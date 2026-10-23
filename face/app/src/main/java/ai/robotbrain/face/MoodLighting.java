package ai.robotbrain.face;

import java.security.GeneralSecurityException;
import java.util.Arrays;
import java.util.Locale;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;

/** Pure emotion mapping and Hao Deng/Zengge Telink packet encoding. */
final class MoodLighting {
    static final int DEFAULT_MESH_UUID = 0x0211;
    static final byte[] DEFAULT_MESH_KEY = "ZenggeMesh".getBytes();
    static final byte[] DEFAULT_MESH_PASSWORD = "ZenggeTechnology".getBytes();

    static final class Profile {
        final String state;
        final int red;
        final int green;
        final int blue;
        final int transitionTenths;

        Profile(String state, int red, int green, int blue, int transitionTenths) {
            this.state = state;
            this.red = red;
            this.green = green;
            this.blue = blue;
            this.transitionTenths = transitionTenths;
        }
    }

    private MoodLighting() { }

    static boolean restingOnDock(boolean docked, String emotion) {
        String state = forState(emotion).state;
        return docked && ("idle".equals(state) || "sleepy".equals(state));
    }

    static Profile forState(String value) {
        String state = value == null ? "idle" : value.toLowerCase(Locale.US);
        // One clear hue per feeling, opposites on opposite sides of the colour wheel (owner 06.10:
        // bright, contrasting colours): joy yellow / sadness deep blue, curiosity cyan / annoyance
        // red, listening blue / speaking green, thinking violet, tenderness pink, calm warm amber.
        switch (state) {
            case "listening": return new Profile(state, 0, 40, 255, 1);       // blue: I hear you
            case "thinking": return new Profile(state, 150, 0, 255, 2);       // violet: working on it
            case "speaking": return new Profile(state, 0, 255, 40, 1);        // green: talking
            case "happy": return new Profile(state, 255, 200, 0, 2);          // sunny yellow
            case "laughing": return new Profile(state, 255, 0, 160, 1);       // hot magenta
            case "curious": return new Profile(state, 0, 230, 255, 2);        // cyan
            case "surprised": return new Profile(state, 255, 255, 255, 0);    // white flash
            case "affectionate": return new Profile(state, 255, 30, 90, 3);   // pink
            case "sleepy": return new Profile(state, 0, 0, 110, 8);           // deep night blue
            case "sad": return new Profile(state, 0, 20, 200, 6);             // deep blue
            case "concerned": return new Profile(state, 255, 100, 0, 1);      // orange
            case "annoyed": return new Profile(state, 255, 0, 0, 1);          // red
            case "error": return new Profile(state, 255, 0, 0, 0);
            default: return new Profile("idle", 255, 110, 30, 6);            // calm warm amber
        }
    }

    static byte[] pairPacket(byte[] sessionRandom) {
        requireLength(sessionRandom, 8, "session random");
        byte[] name = pad16(DEFAULT_MESH_KEY);
        byte[] password = pad16(DEFAULT_MESH_PASSWORD);
        byte[] xor = new byte[16];
        for (int i = 0; i < xor.length; i++) xor[i] = (byte)(name[i] ^ password[i]);
        byte[] encrypted = encrypt(pad16(sessionRandom), xor);
        byte[] packet = new byte[17];
        packet[0] = 0x0c;
        System.arraycopy(sessionRandom, 0, packet, 1, 8);
        System.arraycopy(encrypted, 0, packet, 9, 8);
        return packet;
    }

    static byte[] sessionKey(byte[] sessionRandom, byte[] responseRandom) {
        requireLength(sessionRandom, 8, "session random");
        requireLength(responseRandom, 8, "response random");
        byte[] key = new byte[16];
        byte[] name = pad16(DEFAULT_MESH_KEY);
        byte[] password = pad16(DEFAULT_MESH_PASSWORD);
        for (int i = 0; i < key.length; i++) key[i] = (byte)(name[i] ^ password[i]);
        byte[] random = new byte[16];
        System.arraycopy(sessionRandom, 0, random, 0, 8);
        System.arraycopy(responseRandom, 0, random, 8, 8);
        return encrypt(key, random);
    }

    static byte[] powerPacket(byte[] sessionKey, String address, int destination,
                              boolean on, byte[] sequence) {
        byte[] data = new byte[] {(byte)0xff, 0x01, on ? (byte)0xff : 0,
                0, 0, 0, 0, 0, 0};
        return commandPacket(sessionKey, address, destination, 0xd0, data, sequence);
    }

    static byte[] colorPacket(byte[] sessionKey, String address, int destination,
                              Profile profile, byte[] sequence) {
        int transition = Math.max(0, Math.min(0xffff, profile.transitionTenths));
        byte[] data = new byte[] {(byte)0xff, 0x60,
                (byte)profile.red, (byte)profile.green, (byte)profile.blue,
                0, 0, (byte)(transition & 0xff), (byte)(transition >>> 8)};
        return commandPacket(sessionKey, address, destination, 0xe2, data, sequence);
    }

    static byte[] elkPowerPacket(boolean on) {
        return new byte[] {0x7e, 0x04, 0x04, on ? (byte)0x01 : 0,
                0, on ? (byte)0x01 : 0, (byte)0xff, 0, (byte)0xef};
    }

    static byte[] elkColorPacket(Profile profile) {
        return new byte[] {0x7e, 0x07, 0x05, 0x03,
                (byte)profile.red, (byte)profile.green, (byte)profile.blue,
                0x10, (byte)0xef};
    }

    static byte[] elkBrightnessPacket(int percent) {
        int brightness = Math.max(0, Math.min(100, percent));
        return new byte[] {0x7e, 0x04, 0x01, (byte)brightness,
                (byte)0xff, (byte)0xff, (byte)0xff, 0, (byte)0xef};
    }

    static byte[] elkAdvertisePacket(byte[] command, int sequence, byte host1, byte host2) {
        if (command == null || command.length != 9) {
            throw new IllegalArgumentException("ELK command must be 9 bytes");
        }
        byte[] packet = new byte[25];
        packet[0] = (byte)sequence;
        packet[1] = (byte)0xb0;
        packet[2] = host1;
        packet[3] = host2;
        System.arraycopy(command, 0, packet, 4, command.length);
        int key = 0x7e ^ (sequence & 0xff);
        for (int i = 1; i < packet.length; i++) packet[i] ^= (byte)key;
        return packet;
    }

    static int brightnessForState(String value) {
        String state = value == null ? "idle" : value.toLowerCase(Locale.US);
        // Feelings are clearly visible; calm stays a quiet glow so the strong ones stand out.
        switch (state) {
            case "error": return 85;
            case "surprised": return 85;
            case "laughing": return 80;
            case "listening": return 80;
            case "speaking": return 75;
            case "happy": return 70;
            case "annoyed": return 70;
            case "concerned": return 60;
            case "thinking": return 60;
            case "curious": return 55;
            case "affectionate": return 55;
            case "sad": return 30;
            case "sleepy": return 8;
            default: return 15;
        }
    }

    /** A built-in dynamic mode of the ELK panel, run by the controller itself (smooth, unlike colour writes
     *  that the panel takes about once a second): a feeling shimmers for a few seconds (owner 08.10). */
    static final class Effect {
        final String name;
        final int code;
        final int speed;

        Effect(String name, int code, int speed) {
            this.name = name;
            this.code = code;
            this.speed = speed;
        }
    }

    static Effect effect(String value) {
        String name = value == null ? "" : value.toLowerCase(Locale.US);
        // Codes of the ELK-BLEDOM / Lotus Lantern modes (0x87..0x9c); speed 0..100.
        switch (name) {
            case "rainbow": return new Effect(name, 0x8a, 75);   // seven colours flowing: joy
            case "party": return new Effect(name, 0x88, 90);     // seven colours jumping: laughter, excitement
            case "flash": return new Effect(name, 0x9c, 90);     // white blinking: surprise
            case "warm": return new Effect(name, 0x8e, 70);      // yellow breathing: gladness
            case "love": return new Effect(name, 0x90, 55);      // magenta breathing: tenderness
            case "curious": return new Effect(name, 0x8f, 65);   // cyan breathing: interest
            case "calm": return new Effect(name, 0x8d, 30);      // slow blue breathing
            case "no": return new Effect(name, 0x8b, 70);        // red breathing: «нет»
            case "rgb": return new Effect(name, 0x89, 80);       // three colours flowing: play
            default: return null;
        }
    }

    static byte[] elkEffectPacket(int code) {
        return new byte[] {0x7e, 0x00, 0x03, (byte)code, 0x03, 0, 0, 0, (byte)0xef};
    }

    static byte[] elkSpeedPacket(int percent) {
        int speed = Math.max(0, Math.min(100, percent));
        return new byte[] {0x7e, 0x00, 0x02, (byte)speed, 0, 0, 0, 0, (byte)0xef};
    }

    private static byte[] commandPacket(byte[] sessionKey, String address,
                                        int destination, int command, byte[] data,
                                        byte[] sequence) {
        requireLength(sessionKey, 16, "session key");
        requireLength(sequence, 3, "sequence");
        byte[] mac = macBytes(address);
        reverse(mac);
        byte[] nonce = new byte[] {mac[0], mac[1], mac[2], mac[3], 0x01,
                sequence[0], sequence[1], sequence[2]};
        byte[] payload = new byte[15];
        payload[0] = (byte)(destination & 0xff);
        payload[1] = (byte)(destination >>> 8);
        payload[2] = (byte)command;
        payload[3] = 0x11;
        payload[4] = 0x02;
        System.arraycopy(data, 0, payload, 5, Math.min(data.length, 10));
        byte[] checksum = checksum(sessionKey, nonce, payload);
        byte[] encrypted = crypt(sessionKey, nonce, payload);
        byte[] packet = new byte[20];
        System.arraycopy(sequence, 0, packet, 0, 3);
        System.arraycopy(checksum, 0, packet, 3, 2);
        System.arraycopy(encrypted, 0, packet, 5, 15);
        return packet;
    }

    private static byte[] checksum(byte[] key, byte[] nonce, byte[] payload) {
        byte[] base = new byte[16];
        System.arraycopy(nonce, 0, base, 0, nonce.length);
        base[nonce.length] = (byte)payload.length;
        byte[] check = encrypt(key, base);
        for (int offset = 0; offset < payload.length; offset += 16) {
            byte[] block = new byte[16];
            System.arraycopy(payload, offset, block, 0,
                    Math.min(16, payload.length - offset));
            for (int i = 0; i < 16; i++) check[i] ^= block[i];
            check = encrypt(key, check);
        }
        return check;
    }

    private static byte[] crypt(byte[] key, byte[] nonce, byte[] payload) {
        byte[] result = new byte[payload.length];
        byte[] base = new byte[16];
        System.arraycopy(nonce, 0, base, 1, nonce.length);
        for (int offset = 0; offset < payload.length; offset += 16) {
            byte[] stream = encrypt(key, base);
            int count = Math.min(16, payload.length - offset);
            for (int i = 0; i < count; i++) result[offset + i] =
                    (byte)(stream[i] ^ payload[offset + i]);
            base[0]++;
        }
        return result;
    }

    private static byte[] encrypt(byte[] key, byte[] value) {
        try {
            byte[] reversedKey = Arrays.copyOf(key, 16);
            byte[] reversedValue = Arrays.copyOf(value, 16);
            reverse(reversedKey);
            reverse(reversedValue);
            Cipher cipher = Cipher.getInstance("AES/ECB/NoPadding");
            cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(reversedKey, "AES"));
            byte[] encrypted = cipher.doFinal(reversedValue);
            reverse(encrypted);
            return encrypted;
        } catch (GeneralSecurityException failure) {
            throw new IllegalStateException("AES unavailable", failure);
        }
    }

    private static byte[] pad16(byte[] value) {
        return Arrays.copyOf(value, 16);
    }

    private static byte[] macBytes(String address) {
        String compact = address == null ? "" : address.replace(":", "");
        if (compact.length() != 12) throw new IllegalArgumentException("invalid BLE address");
        byte[] result = new byte[6];
        for (int i = 0; i < result.length; i++) {
            result[i] = (byte)Integer.parseInt(compact.substring(i * 2, i * 2 + 2), 16);
        }
        return result;
    }

    private static void reverse(byte[] value) {
        for (int left = 0, right = value.length - 1; left < right; left++, right--) {
            byte temporary = value[left]; value[left] = value[right]; value[right] = temporary;
        }
    }

    private static void requireLength(byte[] value, int length, String name) {
        if (value == null || value.length != length) {
            throw new IllegalArgumentException(name + " must be " + length + " bytes");
        }
    }

    private static byte[] hex(String value) {
        byte[] result = new byte[value.length() / 2];
        for (int i = 0; i < result.length; i++) result[i] =
                (byte)Integer.parseInt(value.substring(i * 2, i * 2 + 2), 16);
        return result;
    }

    /** Host-runnable protocol vector check: javac MoodLighting.java && java -ea ... */
    public static void main(String[] arguments) {
        for (String state : new String[]{"idle", "sleepy", "listening", "thinking",
                "speaking", "happy", "laughing", "curious", "surprised",
                "affectionate", "sad", "concerned", "annoyed", "error"}) {
            if (restingOnDock(false, state) || restingOnDock(true, state) !=
                    ("idle".equals(state) || "sleepy".equals(state))) {
                throw new AssertionError("dock hides active emotion: " + state);
            }
        }
        byte[] client = hex("0102030405060708");
        byte[] server = hex("1112131415161718");
        byte[] session = sessionKey(client, server);
        if (!Arrays.equals(pairPacket(client), hex("0c0102030405060708b861329b30a51d23")) ||
                !Arrays.equals(session, hex("9d9284addc399b148093ad3ea06f6271")) ||
                !Arrays.equals(colorPacket(session, "AA:BB:CC:DD:EE:FF", 1,
                        new Profile("test", 255, 128, 0, 4), hex("a1b2c3")),
                        hex("a1b2c3e679dcf679fe98ab8814ac78e5df4ae303")) ||
                !Arrays.equals(elkPowerPacket(true), hex("7e0404010001ff00ef")) ||
                !Arrays.equals(elkColorPacket(new Profile("test", 255, 128, 0, 4)),
                        hex("7e070503ff800010ef")) ||
                !Arrays.equals(elkBrightnessPacket(12), hex("7e04010cffffff00ef")) ||
                !Arrays.equals(elkBrightnessPacket(120), hex("7e040164ffffff00ef")) ||
                !Arrays.equals(elkAdvertisePacket(elkPowerPacket(true), 0,
                                (byte)0x12, (byte)0x34),
                        hex("00ce6c4a007a7a7f7e7f817e917e7e7e7e7e7e7e7e7e7e7e7e")) ||
                brightnessForState("idle") != 15 || brightnessForState("error") != 85 ||
                !Arrays.equals(elkEffectPacket(effect("party").code), hex("7e00038803000000ef")) ||
                !Arrays.equals(elkSpeedPacket(150), hex("7e00026400000000ef")) ||
                effect("nonsense") != null || effect("RAINBOW").code != 0x8a) {
            throw new AssertionError("mood light protocol vector mismatch");
        }
        System.out.println("MoodLighting protocol vectors: PASS");
    }
}
