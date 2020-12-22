package com.pjproductions.rest.cryptography;

import com.pjproductions.persistence.storage.data.User;

import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;
import javax.xml.bind.DatatypeConverter;
import java.security.*;
import java.security.spec.InvalidKeySpecException;


public class PasswordCryptography {
    public static final int PASSWORD_MAXSIZE = 300;
    public static final int PASSWORD_MINSIZE = 8;
    private static byte [] PASS_SALT;
    private static final int CRYPTO_ITERATIONS = 200;
    private static final int KEY_LENGTH=96;

    public static boolean isPasswordCorrect(String password, User user){
        return user.getPasswordHash().equals(password);
    }

    public static String passwordToSHA256(String password){

            PBEKeySpec pbeKeySpec = new PBEKeySpec(password.toCharArray(), PASS_SALT, CRYPTO_ITERATIONS, KEY_LENGTH * 8);
        SecretKeyFactory skf = null;
        try {
            skf = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA512");
            return DatatypeConverter.printBase64Binary(skf.generateSecret(pbeKeySpec).getEncoded());
        } catch (NoSuchAlgorithmException | InvalidKeySpecException e) {
            e.printStackTrace();
        }

        return "";
    }

    private static byte[] getSalt() throws NoSuchAlgorithmException, NoSuchProviderException{
        SecureRandom secureRandom = SecureRandom.getInstance("SHA1PRNG", "SUN");

        byte[] salt = new byte[32];
        secureRandom.nextBytes(salt);

        return salt;
    }

    static{
        try {
            PASS_SALT = getSalt();
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        } catch (NoSuchProviderException e) {
            e.printStackTrace();
        }


    }



}
