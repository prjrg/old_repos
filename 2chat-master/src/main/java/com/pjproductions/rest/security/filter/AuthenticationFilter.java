package com.pjproductions.rest.security.filter;

import com.pjproductions.rest.cryptography.token.TokenManager;
import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.PersistenceException;
import org.jose4j.jwt.consumer.InvalidJwtException;

import javax.annotation.Priority;
import javax.ws.rs.Priorities;
import javax.ws.rs.container.ContainerRequestContext;
import javax.ws.rs.container.ContainerRequestFilter;
import javax.ws.rs.core.HttpHeaders;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.SecurityContext;
import javax.ws.rs.ext.Provider;
import java.io.IOException;
import java.security.Principal;


@Secured
@Provider
@Priority(Priorities.AUTHENTICATION)
public class AuthenticationFilter implements ContainerRequestFilter{

    @Override
    public void filter(ContainerRequestContext requestContext) throws IOException {

        // Get the HTTP Authorization header from the request
        String authorizationHeader = requestContext.getHeaderString(HttpHeaders.AUTHORIZATION);

        if(!hasAuthHeader(authorizationHeader)){
            requestContext.abortWith(Response.status(Response.Status.BAD_REQUEST)
                    .entity(OperationResult.CREDENTIALS_REQUIRED.getJSONMessage()).build());
            return;
        }

        // Extract the token from the HTTP Authorization header
        String token = authorizationHeader.substring("Bearer ".length()).trim();

        // Validate the token
        String username = validateToken(token);

        requestContext.setSecurityContext(userSecurityContext(username));
    }


    private boolean hasAuthHeader(String authorizationHeader) {
        return authorizationHeader != null && authorizationHeader.startsWith("Bearer ");
    }

    private String validateToken(String token){

        String username = "";
        try {
            username = TokenManager.validateToken(token);
        } catch (InvalidJwtException e) {
            throw new PersistenceException(OperationResult.INVALID_CREDENTIALS);
        }

        return username;
    }

    private static SecurityContext userSecurityContext(String username){
        return new SecurityContext() {

            @Override
            public Principal getUserPrincipal() {
                return () -> username;
            }

            @Override
            public boolean isUserInRole(String role) {
                return true;
            }

            @Override
            public boolean isSecure() {
                return false;
            }

            @Override
            public String getAuthenticationScheme() {
                return null;
            }
        };
    }



}
