package com.pjproductions.rest.cryptography.token;

import org.jose4j.jwa.AlgorithmConstraints;
import org.jose4j.jwk.RsaJsonWebKey;
import org.jose4j.jwk.RsaJwkGenerator;
import org.jose4j.jws.AlgorithmIdentifiers;
import org.jose4j.jws.JsonWebSignature;
import org.jose4j.jwt.JwtClaims;
import org.jose4j.jwt.consumer.InvalidJwtException;
import org.jose4j.jwt.consumer.JwtConsumer;
import org.jose4j.jwt.consumer.JwtConsumerBuilder;
import org.jose4j.lang.JoseException;

public class TokenManager {
    public static final String TOKEN_EMAIL_KEY = "email";
    public static final String TOKEN_USERNAME_KEY = "username";
    public static final String TOKEN_ID_KEY = "uid";
    private static RsaJsonWebKey rsaJsonWebKey = null;
    private static String ISSUER = "tochat.pjproductions.com";
    private static int timeToExpire = 600;

    static {
        try {
            rsaJsonWebKey = RsaJwkGenerator.generateJwk(2048);
        } catch (JoseException e) {
            e.printStackTrace();
        }
    }

    public static String genToken( String id, String username, String email) {
        // Give the JWK a Key ID (kid), which is just the polite thing to do
        rsaJsonWebKey.setKeyId("k1");

        // Create the Claims, which will be the content from the JWT
        JwtClaims claims = new JwtClaims();
        // who creates the token and signs it
        claims.setIssuer( ISSUER );
        // time when the token will expire (timeToExpire minutes from now)
        claims.setExpirationTimeMinutesInTheFuture( timeToExpire );
        // a unique identifier for the token
        claims.setGeneratedJwtId();
        // when the token was issued/created (now)
        claims.setIssuedAtToNow();
        // time before which the token is not yet valid (1 minutes ago)
        claims.setNotBeforeMinutesInThePast(1);

        claims.setSubject(username);
        claims.setClaim(TOKEN_EMAIL_KEY, email);
        claims.setClaim(TOKEN_USERNAME_KEY, username);
        // transmit the user id for later authentication
        claims.setClaim( TOKEN_ID_KEY, id );

        // A JWT is a JWS and/or a JWE with JSON claims as the payload.
        // In this example it is a JWS so we create a JsonWebSignature object.
        JsonWebSignature jws = new JsonWebSignature();
        // The payload from the JWS is JSON content from the JWT Claims
        jws.setPayload( claims.toJson() );
        // The JWT is signed using the private key
        jws.setKey( rsaJsonWebKey.getPrivateKey() );

        // Set the Key ID (kid) header because it's just the polite thing to do.
        // We only have one key in this example but a using a Key ID helps
        // facilitate a smooth key rollover process
        jws.setKeyIdHeaderValue( rsaJsonWebKey.getKeyId() );

        // Set the signature algorithm on the JWT/JWS that will integrity protect the claims
        jws.setAlgorithmHeaderValue( AlgorithmIdentifiers.RSA_USING_SHA512);

        // Sign the JWS and produce the compact serialization or the complete JWT/JWS
        // representation, which is a string consisting from three dot ('.') separated
        // base64url-encoded parts in the form Header.Payload.Signature
        // If you wanted to encrypt it, you can simply set this jwt as the payload
        // from a JsonWebEncryption object and set the cty (Content Type) header to "jwt".
        try {
            return jws.getCompactSerialization();
        } catch (JoseException e) {
            e.printStackTrace();
            return null;
        }
    }


    public static String validateToken( String jwt ) throws InvalidJwtException {
        JwtConsumer jwtConsumer = new JwtConsumerBuilder()
                // the JWT must have an expiration time
                .setRequireExpirationTime()
                // allow some leeway in validating time based claims to account for clock skew
                .setAllowedClockSkewInSeconds( 1 )
                // whom the JWT needs to have been issued by
                .setExpectedIssuer( ISSUER )
                // verify the signature with the public key
                .setVerificationKey( rsaJsonWebKey.getKey() )
                .setJwsAlgorithmConstraints( // only allow the expected signature algorithm(s) in the given context
                        new AlgorithmConstraints(AlgorithmConstraints.ConstraintType.WHITELIST, //
                                AlgorithmIdentifiers.RSA_USING_SHA512))
                .build();

        //  Validate the JWT and process it to the Claims
        JwtClaims jwtClaims = jwtConsumer.processToClaims( jwt );

        // validate and return the encoded user id
        return jwtClaims.getClaimsMap().get(TOKEN_USERNAME_KEY).toString();
    }

}
