package com.pjproductions.rest.api;

import com.pjproductions.persistence.Persistence;
import com.pjproductions.rest.definition.json.MessageJSON;
import com.pjproductions.rest.definition.json.MessageRequest;
import com.pjproductions.rest.definition.json.ReadMessages;
import com.pjproductions.rest.definition.json.ReadNewMessages;
import com.pjproductions.rest.definition.response.JSONMessage;
import com.pjproductions.rest.security.filter.Secured;

import javax.validation.Valid;
import javax.validation.constraints.NotNull;
import javax.ws.rs.*;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.SecurityContext;
import java.util.Collection;
import java.util.Map;

import static com.pjproductions.rest.definition.OperationResult.OK;
import static com.pjproductions.rest.definition.OperationResult.OK_SENT;

@Path("/message")
@Secured
public class MessagingEndPoint {

    @Context
    SecurityContext securityContext;

    @POST
    @Path("/send1")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    public Response send(@NotNull @Valid MessageRequest<String> request) {
        String username = getUsername();
        Persistence.manager().MESSENGER_CONTROLLER.sendOnePrivate(username, request.to(), request.message());
        return Response.ok(OK_SENT).build();
    }

    @POST
    @Path("/sendmult")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    public Response sendMultiple(@NotNull @Valid MessageRequest<Collection<String>> request) {
        String username = getUsername();
        Persistence.manager().MESSENGER_CONTROLLER.sendMultiplePrivate(username, request.to(), request.message());
        return Response.ok(OK_SENT).build();
    }

    @POST
    @Path("/read")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    public Response readMessages(@NotNull @Valid ReadMessages<String> user){
        String username = getUsername();
        Collection<MessageJSON<String>> msgs = Persistence.manager().MESSENGER_CONTROLLER.readOneUser(username, user);
        return Response.ok(JSONMessage.of(OK, msgs)).build();

    }

    @POST
    @Path("/read/new")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    public Response readNewMessages(@NotNull @Valid ReadNewMessages<String> user){
        String username = getUsername();
        Collection<MessageJSON<String>> msgs = Persistence.manager().MESSENGER_CONTROLLER.readNewOneUser(username, user);
        return Response.ok(JSONMessage.of(OK, msgs)).build();

    }

    @POST
    @Path("/read/several")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    public Response readMessagesMany(@NotNull @Valid Collection<ReadMessages<String>> users){
        String username = getUsername();
        Map<String, Collection<MessageJSON<String>>> msgs = Persistence.manager().MESSENGER_CONTROLLER.readManyUsers(username, users);
        return Response.ok(JSONMessage.of(OK, msgs)).build();
    }

    @POST
    @Path("/read/new/several")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    public Response readNewMessagesMany(@NotNull @Valid Collection<ReadNewMessages<String>> users){
        String username = getUsername();
        Map<String, Collection<MessageJSON<String>>> msgs = Persistence.manager().MESSENGER_CONTROLLER.readNewManyUsers(username, users);
        return Response.ok(JSONMessage.of(OK, msgs)).build();
    }

    @GET
    @Path("/read/all")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response readAllMessages(){
        String username = getUsername();
        Map<String, Collection<MessageJSON<String>>> msgs = Persistence.manager().MESSENGER_CONTROLLER.readAll(username);
        return Response.ok(JSONMessage.of(OK, msgs)).build();
    }

    @GET
    @Path("/check/{username}")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response checkLastId(@PathParam("username") String friend){
        String username = getUsername();

        Long index = Persistence.manager().MESSENGER_CONTROLLER.lastIndex(username, friend);

        return Response.ok(JSONMessage.of(OK, index)).build();
    }

    @POST
    @Path("/check")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    public Response checkLastIdFriends(@NotNull @Valid Collection<String> friends){
        String username = getUsername();
        Map<String, Long> res = Persistence.manager().MESSENGER_CONTROLLER.lastIndex(username, friends);
        return Response.ok(JSONMessage.of(OK, res)).build();
    }

    @GET
    @Path("/check")
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    @Produces(MediaType.APPLICATION_JSON)
    public Response checkLastIdAll(){
        String username = getUsername();
        Map<String, Long> res = Persistence.manager().MESSENGER_CONTROLLER.lastIndex(username);
        return Response.ok(JSONMessage.of(OK, res)).build();
    }


    private String getUsername(){
        return securityContext.getUserPrincipal().getName();
    }
}
