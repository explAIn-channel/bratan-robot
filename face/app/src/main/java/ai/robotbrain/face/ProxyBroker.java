package ai.robotbrain.face;

import android.content.Context;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.system.Os;
import android.util.Base64;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.security.Key;
import java.security.KeyFactory;
import java.security.KeyPairGenerator;
import java.security.KeyStore;
import java.security.MessageDigest;
import java.security.PublicKey;
import java.security.SecureRandom;
import java.security.Signature;
import java.security.interfaces.RSAPublicKey;
import java.security.spec.MGF1ParameterSpec;
import java.security.spec.X509EncodedKeySpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.OAEPParameterSpec;
import javax.crypto.spec.PSource;

/**
 * Local CONNECT proxy whose upstream credentials are encrypted by Android
 * Keystore.  It is intentionally limited to the exact ChatGPT and Microsoft
 * Speech TLS endpoints and has no OAuth,
 * shell, ROS, device, or motion surface.
 */
public final class ProxyBroker {
    static final String PUBLIC_SCHEMA = "robotbrain-phone-proxy-public-key-v1";
    static final String CREDENTIAL_SCHEMA = "robotbrain-phone-proxy-credentials-v1";
    static final String ENVELOPE_SCHEMA = "robotbrain-phone-proxy-provision-v1";
    static final String PROVISIONER_ISSUER = "robotbrain-windows-provisioner-v1";
    private static final String RSA_ALIAS = "robotbrain_proxy_provision_rsa_v1";
    private static final String STORE_ALIAS = "robotbrain_proxy_store_aes_v1";
    private static final String STORE_FILE = "proxy-credentials-v1.enc";
    private static final byte[] ENVELOPE_AAD = ENVELOPE_SCHEMA.getBytes(StandardCharsets.US_ASCII);
    private static final byte[] STORE_AAD = CREDENTIAL_SCHEMA.getBytes(StandardCharsets.US_ASCII);
    private static final Pattern GENERATION = Pattern.compile("[A-Za-z0-9_.:@-]{3,160}");
    private static final Pattern HOST = Pattern.compile("[A-Za-z0-9.-]{1,253}");
    private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");
    private static final Pattern BASE64URL_TOKEN = Pattern.compile("[A-Za-z0-9_-]{43,128}");
    private static final int MAX_ENVELOPE_BYTES = 128 * 1024;
    private static final int MAX_ROUTES = 20;
    private static final long CHALLENGE_LIFETIME_MS = 1_800_000L;
    private static final long MAX_PROVISION_VALIDITY_MS = 1_800_000L;
    private static final long MAX_CLOCK_SKEW_MS = 120_000L;
    private static final String TARGET_ALLOWLIST =
            "speech.platform.bing.com:443";
    private static final Set<String> REQUIRED_CAPABILITIES = Collections.unmodifiableSet(
            new HashSet<>(Arrays.asList(
                    "status.read", "camera.read", "microphone.read",
                    "speaker.write", "face.write", "proxy.read", "proxy.connect")));

    private final Context context;
    private final SecureRandom random = new SecureRandom();
    private final Semaphore connectionCapacity = new Semaphore(4);
    private final AtomicInteger activeTunnels = new AtomicInteger();
    private final List<ProxyListener> listeners = new ArrayList<>();
    private final Set<Socket> generationSockets = Collections.newSetFromMap(
            new java.util.concurrent.ConcurrentHashMap<Socket, Boolean>());
    private volatile long listenerEpoch;
    private byte[] challenge;
    private long challengeExpiresMs;
    private String generation = "";
    private long configVersion;
    private Map<String, byte[]> capabilityHashes = Collections.emptyMap();
    private volatile String lastError = "not_provisioned";

    ProxyBroker(Context context) {
        this.context = context.createDeviceProtectedStorageContext();
        try {
            ensureKeys();
            JSONObject saved = loadStoredCredentials();
            if (saved != null) {
                ParsedCredentials parsed = parseCredentials(saved);
                activate(parsed.routes);
                generation = parsed.generation;
                configVersion = parsed.configVersion;
                capabilityHashes = parsed.capabilityHashes;
                lastError = "";
            }
        } catch (Exception error) {
            stopListeners();
            capabilityHashes = Collections.emptyMap();
            lastError = safeError(error);
        }
    }

    synchronized String publicKeyJson() {
        try {
            ensureKeys();
            long now = System.currentTimeMillis();
            String storedChallenge = context.getSharedPreferences(
                    "proxy-provision", Context.MODE_PRIVATE)
                    .getString("challenge_b64", "");
            long storedExpiry = context.getSharedPreferences(
                    "proxy-provision", Context.MODE_PRIVATE)
                    .getLong("challenge_expires_ms", 0L);
            if (storedExpiry > now && !storedChallenge.isEmpty()) {
                challenge = Base64.decode(storedChallenge, Base64.NO_WRAP);
                challengeExpiresMs = storedExpiry;
            } else {
                challenge = new byte[32];
                random.nextBytes(challenge);
                challengeExpiresMs = now + CHALLENGE_LIFETIME_MS;
                context.getSharedPreferences("proxy-provision", Context.MODE_PRIVATE)
                        .edit()
                        .putString("challenge_b64", Base64.encodeToString(
                                challenge, Base64.NO_WRAP))
                        .putLong("challenge_expires_ms", challengeExpiresMs)
                        .commit();
            }
            byte[] der = keyStore().getCertificate(RSA_ALIAS).getPublicKey().getEncoded();
            JSONObject result = new JSONObject();
            result.put("schema", PUBLIC_SCHEMA);
            result.put("key_algorithm", "RSA-OAEP-SHA256-MGF1-SHA1");
            result.put("public_key_der_b64", Base64.encodeToString(der, Base64.NO_WRAP));
            result.put("key_id", hex(MessageDigest.getInstance("SHA-256").digest(der)));
            result.put("challenge_b64", Base64.encodeToString(challenge, Base64.NO_WRAP));
            result.put("challenge_lifetime_seconds", CHALLENGE_LIFETIME_MS / 1000);
            result.put("current_config_version", configVersion);
            result.put("provisioner_pinned", provisionerPinned());
            return result.toString();
        } catch (Exception error) {
            lastError = safeError(error);
            return errorJson(lastError);
        }
    }

    synchronized ProvisionResult provision(byte[] envelopeBytes) {
        if (envelopeBytes == null || envelopeBytes.length == 0 ||
                envelopeBytes.length > MAX_ENVELOPE_BYTES) {
            return ProvisionResult.failure("envelope_size_invalid");
        }
        if (challenge == null) {
            String saved = context.getSharedPreferences(
                    "proxy-provision", Context.MODE_PRIVATE)
                    .getString("challenge_b64", "");
            challengeExpiresMs = context.getSharedPreferences(
                    "proxy-provision", Context.MODE_PRIVATE)
                    .getLong("challenge_expires_ms", 0L);
            if (!saved.isEmpty()) challenge = Base64.decode(saved, Base64.NO_WRAP);
        }
        if (challenge == null || System.currentTimeMillis() > challengeExpiresMs) {
            return ProvisionResult.failure("challenge_missing_or_expired");
        }
        try {
            JSONObject envelope = new JSONObject(new String(envelopeBytes, StandardCharsets.UTF_8));
            VerifiedEnvelope verified = verifyEnvelope(envelope);
            byte[] wrapped = decodeBounded(envelope, "encrypted_key_b64", 4096);
            byte[] nonce = decodeBounded(envelope, "nonce_b64", 32);
            byte[] ciphertext = decodeBounded(envelope, "ciphertext_b64", MAX_ENVELOPE_BYTES);
            if (nonce.length != 12) throw new IllegalArgumentException("nonce_invalid");

            Cipher unwrap = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding");
            OAEPParameterSpec oaep = new OAEPParameterSpec(
                    "SHA-256", "MGF1", MGF1ParameterSpec.SHA1, PSource.PSpecified.DEFAULT);
            unwrap.init(Cipher.DECRYPT_MODE, keyStore().getKey(RSA_ALIAS, null), oaep);
            byte[] aesBytes = unwrap.doFinal(wrapped);
            if (aesBytes.length != 32) throw new IllegalArgumentException("wrapped_key_invalid");
            byte[] plaintext;
            try {
                javax.crypto.spec.SecretKeySpec temporary =
                        new javax.crypto.spec.SecretKeySpec(aesBytes, "AES");
                Cipher decrypt = Cipher.getInstance("AES/GCM/NoPadding");
                decrypt.init(Cipher.DECRYPT_MODE, temporary, new GCMParameterSpec(128, nonce));
                decrypt.updateAAD(ENVELOPE_AAD);
                plaintext = decrypt.doFinal(ciphertext);
            } finally {
                java.util.Arrays.fill(aesBytes, (byte) 0);
            }
            JSONObject value;
            try {
                value = new JSONObject(new String(plaintext, StandardCharsets.UTF_8));
            } finally {
                java.util.Arrays.fill(plaintext, (byte) 0);
            }
            String expectedChallenge = Base64.encodeToString(challenge, Base64.NO_WRAP);
            if (!expectedChallenge.equals(value.optString("challenge_b64"))) {
                throw new IllegalArgumentException("challenge_mismatch");
            }
            requireBinding(value, "issuer", verified.issuer);
            requireBinding(value, "issuer_key_id", verified.issuerKeyId);
            requireBinding(value, "recipient_key_id", verified.recipientKeyId);
            requireBinding(value, "provision_nonce_b64", verified.provisionNonceB64);
            requireLongBinding(value, "issued_at_ms", verified.issuedAtMs);
            requireLongBinding(value, "expires_at_ms", verified.expiresAtMs);
            requireLongBinding(value, "config_version", verified.configVersion);
            ParsedCredentials parsed = parseCredentials(value);
            List<Route> previous = currentRoutes();
            Map<String, byte[]> previousCapabilities = capabilityHashes;
            String previousGeneration = generation;
            long previousVersion = configVersion;
            stopListeners();
            try {
                startListeners(parsed.routes);
                storeCredentials(value);
            } catch (Exception activationError) {
                stopListeners();
                if (!previous.isEmpty()) startListeners(previous);
                capabilityHashes = previousCapabilities;
                generation = previousGeneration;
                configVersion = previousVersion;
                throw activationError;
            }
            capabilityHashes = parsed.capabilityHashes;
            generation = parsed.generation;
            configVersion = parsed.configVersion;
            lastError = "";
            challenge = null;
            challengeExpiresMs = 0;
            context.getSharedPreferences("proxy-provision", Context.MODE_PRIVATE)
                    .edit().clear().commit();
            return ProvisionResult.success(parsed.routes.size(), generation, configVersion);
        } catch (Exception error) {
            lastError = safeError(error);
            return ProvisionResult.failure(lastError);
        }
    }

    synchronized String healthJson() {
        int ready = 0;
        for (ProxyListener listener : listeners) if (listener.isReady()) ready++;
        JSONObject result = new JSONObject();
        try {
            result.put("schema", "robotbrain-phone-proxy-health-v1");
            result.put("ready", securityConfigured() && ready > 0 && ready == listeners.size());
            result.put("service", "android-keystore-proxy-broker");
            result.put("loopback_only", true);
            result.put("target_allowlist", TARGET_ALLOWLIST);
            result.put("keystore_backed", true);
            result.put("plaintext_persisted", false);
            result.put("route_count", listeners.size());
            result.put("listener_count", ready);
            result.put("active_tunnel_count", activeTunnels.get());
            result.put("generation", generation);
            result.put("config_version", configVersion);
            result.put("capability_auth", !capabilityHashes.isEmpty());
            result.put("authorization_mode", capabilityHashes.isEmpty() ?
                    "unconfigured_fail_closed" : "scoped_capabilities");
            result.put("provisioner_pinned", provisionerPinned());
            result.put("last_error", lastError);
        } catch (JSONException ignored) { }
        return result.toString();
    }

    synchronized void shutdown() {
        stopListeners();
    }

    synchronized boolean securityConfigured() {
        // Missing, corrupt and legacy stores never authorize local callers.
        return configVersion > 0 && capabilityHashes.keySet().equals(REQUIRED_CAPABILITIES);
    }

    synchronized boolean ready() {
        if (!securityConfigured() || listeners.isEmpty()) return false;
        for (ProxyListener listener : listeners) if (!listener.isReady()) return false;
        return true;
    }

    boolean isNetworkActive() {
        return activeTunnels.get() > 0;
    }

    synchronized boolean authorize(String scope, String authorization) {
        if (!securityConfigured()) return false;
        byte[] expected = capabilityHashes.get(scope);
        if (expected == null || authorization == null ||
                !authorization.startsWith("Bearer ")) return false;
        String token = authorization.substring(7);
        if (!BASE64URL_TOKEN.matcher(token).matches()) return false;
        try {
            byte[] supplied = MessageDigest.getInstance("SHA-256").digest(
                    token.getBytes(StandardCharsets.US_ASCII));
            return MessageDigest.isEqual(expected, supplied);
        } catch (Exception ignored) {
            return false;
        }
    }

    private void ensureKeys() throws Exception {
        KeyStore store = keyStore();
        if (!store.containsAlias(RSA_ALIAS)) {
            KeyPairGenerator generator = KeyPairGenerator.getInstance(
                    KeyProperties.KEY_ALGORITHM_RSA, "AndroidKeyStore");
            generator.initialize(new KeyGenParameterSpec.Builder(
                    RSA_ALIAS, KeyProperties.PURPOSE_DECRYPT)
                    .setKeySize(2048)
                    .setDigests(KeyProperties.DIGEST_SHA256, KeyProperties.DIGEST_SHA1)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_OAEP)
                    .build());
            generator.generateKeyPair();
        }
        if (!store.containsAlias(STORE_ALIAS)) {
            KeyGenerator generator = KeyGenerator.getInstance(
                    KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
            generator.init(new KeyGenParameterSpec.Builder(
                    STORE_ALIAS, KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                    .setRandomizedEncryptionRequired(true)
                    .build());
            generator.generateKey();
        }
    }

    private KeyStore keyStore() throws Exception {
        KeyStore store = KeyStore.getInstance("AndroidKeyStore");
        store.load(null);
        return store;
    }

    private boolean provisionerPinned() {
        try {
            return pinnedProvisionerKey() != null;
        } catch (Exception ignored) {
            return false;
        }
    }

    private PublicKey pinnedProvisionerKey() throws Exception {
        String encoded = BuildConfig.ROBOTBRAIN_PROVISIONER_PUBLIC_KEY_DER_B64;
        if (encoded == null || encoded.isEmpty()) {
            throw new IllegalArgumentException("provisioner_key_unconfigured");
        }
        byte[] der = Base64.decode(encoded, Base64.NO_WRAP);
        if (der.length < 384 || der.length > 2048) {
            throw new IllegalArgumentException("provisioner_key_invalid");
        }
        PublicKey key = KeyFactory.getInstance("RSA").generatePublic(
                new X509EncodedKeySpec(der));
        if (!(key instanceof RSAPublicKey) ||
                ((RSAPublicKey) key).getModulus().bitLength() < 3072) {
            throw new IllegalArgumentException("provisioner_key_invalid");
        }
        return key;
    }

    private VerifiedEnvelope verifyEnvelope(JSONObject envelope) throws Exception {
        Set<String> expectedFields = new HashSet<>(Arrays.asList(
                "schema", "issuer", "issuer_key_id", "recipient_key_id",
                "issued_at_ms", "expires_at_ms", "config_version",
                "provision_nonce_b64", "encrypted_key_b64", "nonce_b64",
                "ciphertext_b64", "signature_b64"));
        Set<String> actualFields = new HashSet<>();
        JSONArray names = envelope.names();
        if (names != null) {
            for (int index = 0; index < names.length(); index++) {
                actualFields.add(names.getString(index));
            }
        }
        if (!actualFields.equals(expectedFields) ||
                !ENVELOPE_SCHEMA.equals(envelope.optString("schema"))) {
            throw new IllegalArgumentException("envelope_schema_invalid");
        }
        String issuer = envelope.getString("issuer");
        String issuerKeyId = envelope.getString("issuer_key_id");
        String recipientKeyId = envelope.getString("recipient_key_id");
        String provisionNonce = envelope.getString("provision_nonce_b64");
        long issuedAtMs = strictPositiveLong(envelope, "issued_at_ms");
        long expiresAtMs = strictPositiveLong(envelope, "expires_at_ms");
        long nextVersion = strictPositiveLong(envelope, "config_version");
        if (!PROVISIONER_ISSUER.equals(issuer)) {
            throw new IllegalArgumentException("provisioner_issuer_invalid");
        }
        PublicKey provisioner = pinnedProvisionerKey();
        String pinnedKeyId = hex(MessageDigest.getInstance("SHA-256").digest(
                provisioner.getEncoded()));
        if (!SHA256.matcher(issuerKeyId).matches() ||
                !MessageDigest.isEqual(pinnedKeyId.getBytes(StandardCharsets.US_ASCII),
                        issuerKeyId.getBytes(StandardCharsets.US_ASCII))) {
            throw new IllegalArgumentException("provisioner_key_id_invalid");
        }
        byte[] deviceDer = keyStore().getCertificate(RSA_ALIAS).getPublicKey().getEncoded();
        String deviceKeyId = hex(MessageDigest.getInstance("SHA-256").digest(deviceDer));
        if (!SHA256.matcher(recipientKeyId).matches() ||
                !MessageDigest.isEqual(deviceKeyId.getBytes(StandardCharsets.US_ASCII),
                        recipientKeyId.getBytes(StandardCharsets.US_ASCII))) {
            throw new IllegalArgumentException("recipient_key_id_invalid");
        }
        byte[] provisionNonceBytes;
        try {
            provisionNonceBytes = Base64.decode(provisionNonce, Base64.NO_WRAP);
        } catch (IllegalArgumentException error) {
            throw new IllegalArgumentException("provision_nonce_invalid", error);
        }
        if (provisionNonceBytes.length != 32) {
            throw new IllegalArgumentException("provision_nonce_invalid");
        }
        long now = System.currentTimeMillis();
        if (issuedAtMs > now + MAX_CLOCK_SKEW_MS) {
            throw new IllegalArgumentException("provision_issued_in_future");
        }
        if (expiresAtMs <= now || expiresAtMs <= issuedAtMs ||
                expiresAtMs - issuedAtMs > MAX_PROVISION_VALIDITY_MS) {
            throw new IllegalArgumentException("provision_expired_or_ttl_invalid");
        }
        if (nextVersion <= configVersion) {
            throw new IllegalArgumentException("config_version_not_monotonic");
        }
        byte[] signatureBytes = decodeBounded(envelope, "signature_b64", 1024);
        Signature signature = Signature.getInstance("SHA256withRSA");
        signature.initVerify(provisioner);
        signature.update(canonicalEnvelope(envelope));
        if (!signature.verify(signatureBytes)) {
            throw new IllegalArgumentException("provision_signature_invalid");
        }
        return new VerifiedEnvelope(issuer, issuerKeyId, recipientKeyId,
                provisionNonce, issuedAtMs, expiresAtMs, nextVersion);
    }

    private static byte[] canonicalEnvelope(JSONObject envelope) throws JSONException {
        String[] fields = {"schema", "issuer", "issuer_key_id", "recipient_key_id",
                "issued_at_ms", "expires_at_ms", "config_version",
                "provision_nonce_b64", "encrypted_key_b64", "nonce_b64",
                "ciphertext_b64"};
        ByteArrayOutputStream output = new ByteArrayOutputStream(4096);
        try {
            for (String field : fields) {
                String value = envelope.get(field).toString();
                byte[] encoded = value.getBytes(StandardCharsets.UTF_8);
                output.write(field.getBytes(StandardCharsets.US_ASCII));
                output.write('=');
                output.write(Integer.toString(encoded.length).getBytes(StandardCharsets.US_ASCII));
                output.write(':');
                output.write(encoded);
                output.write('\n');
            }
        } catch (IOException impossible) {
            throw new IllegalStateException(impossible);
        }
        return output.toByteArray();
    }

    private static long strictPositiveLong(JSONObject value, String key) {
        Object raw = value.opt(key);
        if (!(raw instanceof Long) && !(raw instanceof Integer)) {
            throw new IllegalArgumentException(key + "_invalid");
        }
        long result = ((Number) raw).longValue();
        if (result <= 0) throw new IllegalArgumentException(key + "_invalid");
        return result;
    }

    private static void requireBinding(JSONObject value, String key, String expected) {
        if (!expected.equals(value.optString(key))) {
            throw new IllegalArgumentException(key + "_binding_mismatch");
        }
    }

    private static void requireLongBinding(JSONObject value, String key, long expected) {
        if (strictPositiveLong(value, key) != expected) {
            throw new IllegalArgumentException(key + "_binding_mismatch");
        }
    }

    private ParsedCredentials parseCredentials(JSONObject value) throws JSONException {
        Set<String> modernFields = new HashSet<>(Arrays.asList(
                "schema", "generation", "capabilities", "routes",
                "challenge_b64", "issuer", "issuer_key_id", "recipient_key_id",
                "issued_at_ms", "expires_at_ms", "config_version",
                "provision_nonce_b64"));
        Set<String> actualFields = jsonFields(value);
        // The deployed pre-capability envelope also carries its one-shot
        // challenge/binding metadata. Its parser intentionally consumed only
        // schema, generation and routes, so preserve that forward-compatible
        // shape instead of requiring an artificial three-field equality.
        if (!actualFields.equals(modernFields) ||
                !CREDENTIAL_SCHEMA.equals(value.optString("schema"))) {
            throw new IllegalArgumentException("credential_schema_invalid");
        }
        String nextGeneration = value.optString("generation");
        if (!GENERATION.matcher(nextGeneration).matches()) {
            throw new IllegalArgumentException("generation_invalid");
        }
        Map<String, byte[]> hashes = new HashMap<>();
        long nextVersion = 0L;
        {
            nextVersion = strictPositiveLong(value, "config_version");
            JSONObject capabilities = value.optJSONObject("capabilities");
            if (capabilities == null) {
                throw new IllegalArgumentException("capabilities_missing");
            }
            Set<String> capabilityNames = new HashSet<>();
            JSONArray capabilityKeys = capabilities.names();
            if (capabilityKeys != null) {
                for (int index = 0; index < capabilityKeys.length(); index++) {
                    capabilityNames.add(capabilityKeys.getString(index));
                }
            }
            if (!capabilityNames.equals(REQUIRED_CAPABILITIES)) {
                throw new IllegalArgumentException("capability_set_invalid");
            }
            for (String scope : REQUIRED_CAPABILITIES) {
                String digest = capabilities.optString(scope);
                if (!SHA256.matcher(digest).matches()) {
                    throw new IllegalArgumentException("capability_hash_invalid");
                }
                hashes.put(scope, decodeHex(digest));
            }
        }
        JSONArray values = value.optJSONArray("routes");
        if (values == null || values.length() < 1 || values.length() > MAX_ROUTES) {
            throw new IllegalArgumentException("route_count_invalid");
        }
        List<Route> result = new ArrayList<>();
        Set<Integer> ports = new HashSet<>();
        Set<String> slots = new HashSet<>();
        for (int index = 0; index < values.length(); index++) {
            JSONObject route = values.getJSONObject(index);
            if (!jsonFields(route).equals(new HashSet<>(Arrays.asList(
                    "slot", "local_port", "upstream_host", "upstream_port",
                    "username", "password")))) {
                throw new IllegalArgumentException("route_schema_invalid");
            }
            int localPort = route.optInt("local_port", -1);
            int upstreamPort = route.optInt("upstream_port", -1);
            String slot = route.optString("slot");
            String host = route.optString("upstream_host").toLowerCase(Locale.US);
            String username = route.optString("username");
            String password = route.optString("password");
            if (localPort < 19000 || localPort > 19999 ||
                    upstreamPort < 1 || upstreamPort > 65535 ||
                    !GENERATION.matcher(slot).matches() ||
                    !HOST.matcher(host).matches() || host.startsWith(".") || host.endsWith(".") ||
                    username.isEmpty() || username.length() > 512 ||
                    password.isEmpty() || password.length() > 512 ||
                    !ports.add(localPort) || !slots.add(slot)) {
                throw new IllegalArgumentException("route_invalid");
            }
            result.add(new Route(slot, localPort, host, upstreamPort, username, password));
        }
        return new ParsedCredentials(result, hashes, nextGeneration, nextVersion);
    }

    private static Set<String> jsonFields(JSONObject value) throws JSONException {
        Set<String> fields = new HashSet<>();
        JSONArray names = value.names();
        if (names != null) {
            for (int index = 0; index < names.length(); index++) {
                fields.add(names.getString(index));
            }
        }
        return fields;
    }

    private void storeCredentials(JSONObject value) throws Exception {
        SecretKey key = (SecretKey) keyStore().getKey(STORE_ALIAS, null);
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, key);
        cipher.updateAAD(STORE_AAD);
        byte[] ciphertext = cipher.doFinal(value.toString().getBytes(StandardCharsets.UTF_8));
        JSONObject stored = new JSONObject();
        stored.put("schema", "robotbrain-phone-proxy-keystore-store-v1");
        stored.put("nonce_b64", Base64.encodeToString(cipher.getIV(), Base64.NO_WRAP));
        stored.put("ciphertext_b64", Base64.encodeToString(ciphertext, Base64.NO_WRAP));
        File destination = new File(context.getFilesDir(), STORE_FILE);
        File temporary = new File(context.getFilesDir(), STORE_FILE + ".tmp");
        try (FileOutputStream output = new FileOutputStream(temporary, false)) {
            output.write(stored.toString().getBytes(StandardCharsets.UTF_8));
            output.flush();
            output.getFD().sync();
        }
        Os.rename(temporary.getAbsolutePath(), destination.getAbsolutePath());
    }

    private JSONObject loadStoredCredentials() throws Exception {
        File source = new File(context.getFilesDir(), STORE_FILE);
        if (!source.isFile()) return null;
        byte[] raw;
        try (FileInputStream input = new FileInputStream(source)) {
            raw = readBounded(input, MAX_ENVELOPE_BYTES);
        }
        JSONObject stored = new JSONObject(new String(raw, StandardCharsets.UTF_8));
        if (!"robotbrain-phone-proxy-keystore-store-v1".equals(stored.optString("schema"))) {
            throw new IllegalArgumentException("credential_store_schema_invalid");
        }
        byte[] nonce = decodeBounded(stored, "nonce_b64", 32);
        byte[] ciphertext = decodeBounded(stored, "ciphertext_b64", MAX_ENVELOPE_BYTES);
        SecretKey key = (SecretKey) keyStore().getKey(STORE_ALIAS, null);
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.DECRYPT_MODE, key, new GCMParameterSpec(128, nonce));
        cipher.updateAAD(STORE_AAD);
        byte[] plaintext = cipher.doFinal(ciphertext);
        JSONObject value = new JSONObject(new String(plaintext, StandardCharsets.UTF_8));
        java.util.Arrays.fill(plaintext, (byte) 0);
        return value;
    }

    private synchronized void activate(List<Route> routes) throws IOException {
        stopListeners();
        startListeners(routes);
    }

    private void startListeners(List<Route> routes) throws IOException {
        List<ProxyListener> started = new ArrayList<>();
        try {
            for (Route route : routes) {
                ProxyListener listener = new ProxyListener(route);
                listener.bind();
                listener.start();
                started.add(listener);
            }
            listeners.addAll(started);
        } catch (IOException error) {
            for (ProxyListener listener : started) listener.shutdown();
            throw error;
        }
    }

    private List<Route> currentRoutes() {
        List<Route> result = new ArrayList<>();
        for (ProxyListener listener : listeners) result.add(listener.route);
        return result;
    }

    private void stopListeners() {
        ++listenerEpoch;
        for (ProxyListener listener : listeners) listener.shutdown();
        listeners.clear();
        for (Socket socket : generationSockets) {
            try { socket.close(); } catch (IOException ignored) { }
        }
    }

    private final class ProxyListener extends Thread {
        final Route route;
        final long epoch = listenerEpoch;
        private volatile boolean running = true;
        private ServerSocket server;

        ProxyListener(Route route) {
            super("robotbrain-proxy-" + route.localPort);
            this.route = route;
            setDaemon(true);
        }

        void bind() throws IOException {
            server = new ServerSocket();
            server.setReuseAddress(true);
            server.bind(new InetSocketAddress(
                    InetAddress.getByName("127.0.0.1"), route.localPort), 4);
        }

        boolean isReady() { return running && server != null && server.isBound() && !server.isClosed(); }

        @Override public void run() {
            while (running) {
                try {
                    Socket client = server.accept();
                    generationSockets.add(client);
                    if (epoch != listenerEpoch) { client.close(); generationSockets.remove(client); continue; }
                    if (!client.getInetAddress().isLoopbackAddress() ||
                            !connectionCapacity.tryAcquire()) {
                        client.close();
                        generationSockets.remove(client);
                        continue;
                    }
                    Thread worker = new Thread(() -> {
                        try { handleTunnel(client, route, epoch); }
                        finally {
                            try { client.close(); } catch (IOException ignored) { }
                            generationSockets.remove(client);
                            connectionCapacity.release();
                        }
                    }, "robotbrain-proxy-connection");
                    worker.setDaemon(true);
                    worker.start();
                } catch (IOException error) {
                    if (running) lastError = "listener_io_error";
                }
            }
        }

        void shutdown() {
            running = false;
            if (server != null) try { server.close(); } catch (IOException ignored) { }
        }
    }

    private void handleTunnel(Socket client, Route route, long epoch) {
        boolean activityCounted = false;
        try (Socket upstream = new Socket()) {
            generationSockets.add(upstream);
            try {
            if (epoch != listenerEpoch) return;
            client.setSoTimeout(1_000);
            byte[] request = readHeader(client.getInputStream(), 16 * 1024,
                    android.os.SystemClock.elapsedRealtime()+15_000L);
            String head = new String(request, StandardCharsets.US_ASCII);
            String first = head.substring(0, head.indexOf("\r\n"));
            String[] requestLine = first.split(" ", -1);
            if (requestLine.length != 3 || !"CONNECT".equals(requestLine[0]) ||
                    !("HTTP/1.1".equals(requestLine[2]) ||
                            "HTTP/1.0".equals(requestLine[2])) ||
                    !("speech.platform.bing.com:443".equals(requestLine[1]) ||
                      "api.openai.com:443".equals(requestLine[1])) ||
                    !authorizeProxyHeader(head)) {
                writeStatus(client, 403, "Forbidden");
                return;
            }
            String target = requestLine[1];
            if (epoch != listenerEpoch) return;
            upstream.connect(new InetSocketAddress(route.upstreamHost, route.upstreamPort), 10_000);
            if (epoch != listenerEpoch) return;
            upstream.setSoTimeout(1_000);
            String authorization = Base64.encodeToString(
                    (route.username + ":" + route.password).getBytes(StandardCharsets.UTF_8),
                    Base64.NO_WRAP);
            String connect = first + "\r\nHost: " + target + "\r\n" +
                    "Proxy-Authorization: Basic " + authorization + "\r\n" +
                    "Proxy-Connection: Keep-Alive\r\n\r\n";
            OutputStream upstreamOut = upstream.getOutputStream();
            upstreamOut.write(connect.getBytes(StandardCharsets.US_ASCII));
            upstreamOut.flush();
            byte[] response = readHeader(upstream.getInputStream(), 16 * 1024,
                    android.os.SystemClock.elapsedRealtime()+15_000L);
            client.getOutputStream().write(response);
            client.getOutputStream().flush();
            String status = new String(response, StandardCharsets.US_ASCII)
                    .split("\r\n", 2)[0];
            if (!status.matches("HTTP/1\\.[01] 2[0-9][0-9].*")) return;
            if (epoch != listenerEpoch) return;
            activeTunnels.incrementAndGet();
            activityCounted = true;
            client.setSoTimeout(120_000);
            upstream.setSoTimeout(120_000);
            Thread reverse = new Thread(() -> copyQuietly(upstream, client),
                    "robotbrain-proxy-reverse");
            reverse.setDaemon(true);
            reverse.start();
            copyQuietly(client, upstream);
            try { reverse.join(2000); } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            } finally { generationSockets.remove(upstream); }
        } catch (Exception ignored) {
        } finally {
            if (activityCounted) activeTunnels.decrementAndGet();
        }
    }

    private boolean authorizeProxyHeader(String head) {
        String supplied = null;
        for (String line : head.split("\r\n")) {
            int colon = line.indexOf(':');
            if (colon <= 0 || !"Proxy-Authorization".equalsIgnoreCase(
                    line.substring(0, colon).trim())) continue;
            if (supplied != null) return false;
            supplied = line.substring(colon + 1).trim();
        }
        if (supplied == null || !supplied.startsWith("Basic ")) return false;
        try {
            byte[] decoded = Base64.decode(supplied.substring(6), Base64.NO_WRAP);
            String credential = new String(decoded, StandardCharsets.UTF_8);
            int colon = credential.indexOf(':');
            return colon > 0 && "robotbrain".equals(credential.substring(0, colon)) &&
                    authorize("proxy.connect", "Bearer " + credential.substring(colon + 1));
        } catch (IllegalArgumentException ignored) {
            return false;
        }
    }

    private static void copyQuietly(Socket source, Socket destination) {
        byte[] buffer = new byte[16 * 1024];
        try {
            InputStream input = source.getInputStream();
            OutputStream output = destination.getOutputStream();
            int count;
            while ((count = input.read(buffer)) >= 0) {
                if (count == 0) continue;
                output.write(buffer, 0, count);
                output.flush();
            }
        } catch (IOException ignored) { }
        try { destination.shutdownOutput(); } catch (IOException ignored) { }
    }

    private static byte[] readHeader(InputStream input, int maximum, long deadline) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream(512);
        int state = 0;
        while (output.size() < maximum) {
            if (android.os.SystemClock.elapsedRealtime() >= deadline)
                throw new IOException("header_deadline");
            int value;
            try {
                value = input.read();
            } catch (java.net.SocketTimeoutException timeout) {
                continue;
            }
            if (value < 0) throw new IOException("header_incomplete");
            output.write(value);
            if ((state == 0 || state == 2) && value == '\r') state++;
            else if ((state == 1 || state == 3) && value == '\n') state++;
            else state = value == '\r' ? 1 : 0;
            if (state == 4) return output.toByteArray();
        }
        throw new IOException("header_too_large");
    }

    private static byte[] readBounded(InputStream input, int maximum) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int count;
        while ((count = input.read(buffer)) >= 0) {
            output.write(buffer, 0, count);
            if (output.size() > maximum) throw new IOException("file_too_large");
        }
        return output.toByteArray();
    }

    private static byte[] decodeBounded(JSONObject value, String key, int maximum) {
        try {
            byte[] decoded = Base64.decode(value.getString(key), Base64.NO_WRAP);
            if (decoded.length == 0 || decoded.length > maximum) {
                throw new IllegalArgumentException(key + "_size_invalid");
            }
            return decoded;
        } catch (JSONException | IllegalArgumentException error) {
            throw new IllegalArgumentException(key + "_invalid", error);
        }
    }

    private static void writeStatus(Socket socket, int status, String reason) {
        try {
            socket.getOutputStream().write(("HTTP/1.1 " + status + " " + reason +
                    "\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
                    .getBytes(StandardCharsets.US_ASCII));
        } catch (IOException ignored) { }
    }

    private static String safeError(Exception error) {
        String message = error.getMessage();
        if (message != null && message.matches("[A-Za-z0-9_.:-]{1,120}")) return message;
        return error.getClass().getSimpleName();
    }

    private static String errorJson(String error) {
        try {
            return new JSONObject().put("ok", false).put("error", error).toString();
        } catch (JSONException ignored) { return "{\"ok\":false}"; }
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) result.append(String.format(Locale.US, "%02x", value));
        return result.toString();
    }

    private static byte[] decodeHex(String value) {
        byte[] result = new byte[value.length() / 2];
        for (int index = 0; index < result.length; index++) {
            int high = Character.digit(value.charAt(index * 2), 16);
            int low = Character.digit(value.charAt(index * 2 + 1), 16);
            if (high < 0 || low < 0) throw new IllegalArgumentException("hex_invalid");
            result[index] = (byte) ((high << 4) | low);
        }
        return result;
    }

    private static final class VerifiedEnvelope {
        final String issuer;
        final String issuerKeyId;
        final String recipientKeyId;
        final String provisionNonceB64;
        final long issuedAtMs;
        final long expiresAtMs;
        final long configVersion;

        VerifiedEnvelope(String issuer, String issuerKeyId, String recipientKeyId,
                         String provisionNonceB64, long issuedAtMs, long expiresAtMs,
                         long configVersion) {
            this.issuer = issuer;
            this.issuerKeyId = issuerKeyId;
            this.recipientKeyId = recipientKeyId;
            this.provisionNonceB64 = provisionNonceB64;
            this.issuedAtMs = issuedAtMs;
            this.expiresAtMs = expiresAtMs;
            this.configVersion = configVersion;
        }
    }

    private static final class ParsedCredentials {
        final List<Route> routes;
        final Map<String, byte[]> capabilityHashes;
        final String generation;
        final long configVersion;

        ParsedCredentials(List<Route> routes, Map<String, byte[]> capabilityHashes,
                          String generation, long configVersion) {
            this.routes = routes;
            this.capabilityHashes = Collections.unmodifiableMap(
                    new HashMap<>(capabilityHashes));
            this.generation = generation;
            this.configVersion = configVersion;
        }
    }

    private static final class Route {
        final String slot;
        final int localPort;
        final String upstreamHost;
        final int upstreamPort;
        final String username;
        final String password;

        Route(String slot, int localPort, String upstreamHost, int upstreamPort,
              String username, String password) {
            this.slot = slot;
            this.localPort = localPort;
            this.upstreamHost = upstreamHost;
            this.upstreamPort = upstreamPort;
            this.username = username;
            this.password = password;
        }
    }

    static final class ProvisionResult {
        final boolean ok;
        final int routeCount;
        final String generation;
        final long configVersion;
        final String error;

        private ProvisionResult(boolean ok, int routeCount, String generation,
                                long configVersion, String error) {
            this.ok = ok;
            this.routeCount = routeCount;
            this.generation = generation;
            this.configVersion = configVersion;
            this.error = error;
        }

        static ProvisionResult success(int count, String generation, long configVersion) {
            return new ProvisionResult(true, count, generation, configVersion, "");
        }

        static ProvisionResult failure(String error) {
            return new ProvisionResult(false, 0, "", 0, error);
        }

        String json() {
            try {
                return new JSONObject().put("ok", ok).put("route_count", routeCount)
                        .put("generation", generation).put("config_version", configVersion)
                        .put("error", error).toString();
            } catch (JSONException ignored) { return "{\"ok\":false}"; }
        }
    }
}
