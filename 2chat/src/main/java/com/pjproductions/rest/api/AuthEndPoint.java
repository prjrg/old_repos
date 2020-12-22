package com.pjproductions.rest.api;

import com.pjproductions.persistence.Persistence;
import com.pjproductions.rest.cryptography.PasswordCryptography;
import com.pjproductions.rest.cryptography.token.TokenManager;
import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.persistence.storage.data.User;
import com.pjproductions.rest.definition.response.JSONMessage;
import com.pjproductions.rest.exception.PersistenceException;
import com.pjproductions.rest.exception.request.AuthExceptionChat;
import com.pjproductions.rest.security.validation.PreCheck;
import com.pjproductions.type.Pair;

import javax.validation.constraints.NotNull;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.xml.bind.DatatypeConverter;

import static com.pjproductions.rest.definition.OperationResult.*;
import static javax.ws.rs.core.HttpHeaders.AUTHORIZATION;

@Path("/auth")
public class AuthEndPoint {

    @POST
    @Path("/login")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response login(@NotNull @HeaderParam(AUTHORIZATION) String authProperty){
        Pair<String, String> credentials = validateLogin(authProperty);

        String username = credentials.getFirst();
        String password = credentials.getSecond();

        User user = Persistence.manager().USERS_CONTROLLER.isPasswordCorrect(username, password);
        if(!user.getPasswordHash().isEmpty()) throw new PersistenceException(WRONG_PASSWORD);

        String token = TokenManager.genToken(user.getId().toString(), user.getUsername(), user.getEmail());

        return Response.ok(JSONMessage.of(OK, token)).build();
    }

    @POST
    @Path("/register")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    @PreCheck(params = {"username", "email", "password", "password2"})
    public Response register(@FormParam("username") String username, @FormParam("email") String email,
                             @FormParam("password") String password, @FormParam("password2") String password2) {


        String pass = PasswordCryptography.passwordToSHA256(password);
        Persistence.manager().USERS_CONTROLLER.addUser(username, email, pass);

        return Response.ok(OK_CREATED).build();

    }

    private static Pair<String, String> validateLogin(String credentials) throws AuthExceptionChat {

        credentials = credentials.replaceFirst("[B|b]asic ", "");
        byte[] decodedCredentialsBytes = DatatypeConverter.parseBase64Binary(credentials);
        if(decodedCredentialsBytes == null || decodedCredentialsBytes.length == 0) throw new AuthExceptionChat(OperationResult.INVALID_OPERATION);
        String[] res = new String(decodedCredentialsBytes).split(":", 2);
        if(res.length != 2) throw new AuthExceptionChat(OperationResult.CREDENTIALS_REQUIRED);
        res[1] = PasswordCryptography.passwordToSHA256(res[1]);

        if(res[1].isEmpty()) throw new AuthExceptionChat(OperationResult.INVALID_CREDENTIALS);

        return Pair.of(res[0], res[1]);
    }



}
