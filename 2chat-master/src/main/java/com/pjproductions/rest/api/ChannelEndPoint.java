package com.pjproductions.rest.api;

import com.pjproductions.rest.security.filter.Secured;

import javax.validation.constraints.NotNull;
import javax.ws.rs.*;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.SecurityContext;

@Path("/channels")
@Secured
public class ChannelEndPoint {

    @Context
    SecurityContext securityContext;

    @POST
    @Path("/{username}/send")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response send(@NotNull @FormParam("message") String message) {
        String username = getUsername();

        return Response.ok().build();
    }

    private String getUsername(){
        return securityContext.getUserPrincipal().getName();
    }
}
