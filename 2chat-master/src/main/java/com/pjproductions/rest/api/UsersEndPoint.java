package com.pjproductions.rest.api;


import com.pjproductions.persistence.Persistence;
import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.definition.json.Friend;
import com.pjproductions.rest.definition.response.JSONMessage;
import com.pjproductions.rest.security.filter.Secured;

import javax.validation.constraints.NotNull;
import javax.ws.rs.*;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.SecurityContext;
import java.util.Collection;

import static com.pjproductions.rest.definition.OperationResult.OK;


@Path("/user")
@Secured
public class UsersEndPoint {

    @Context
    SecurityContext securityContext;

    @GET
    @Path("/friends")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response listFriends() {
        String username = getUsername();

        Collection<Friend> friends = Persistence.manager().USERS_CONTROLLER.userFriends(username);
        return Response.ok(JSONMessage.of(OK, friends)).build();
    }

    @GET
    @Path("/{username}/add")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response addFriend(@NotNull @PathParam("username") String friend) {

        String username = getUsername();
        Persistence.manager().USERS_CONTROLLER.sendRequest(username, friend);

        return Response.ok(OperationResult.OK_ADDED).build();
    }

    @GET
    @Path("/{username}/remove")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response removeFriend(@NotNull @PathParam("username") String friend) {
        String username = getUsername();

        Persistence.manager().USERS_CONTROLLER.removeFriend(username, friend);

        return Response.ok(OperationResult.OK).build();
    }

    @GET
    @Path("/{username}/block")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response blockUser(@NotNull @PathParam("username") String user){

        String username = getUsername();
        Persistence.manager().USERS_CONTROLLER.blockUser(username, user);

        return Response.ok(OperationResult.OK).build();
    }

    @GET
    @Path("/requests")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response requests(){
        String username = getUsername();

        Collection<String> response = Persistence.manager().USERS_CONTROLLER.friendRequests(username);

        return Response.ok(JSONMessage.of(OK, response)).build();
    }

    @GET
    @Path("/{username}/request/{accept}")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
    public Response acceptRequest(@PathParam("username") String user, @PathParam("accept") boolean accept){
        String username = getUsername();

        Persistence.manager().USERS_CONTROLLER.handleFriendRequest(username, user, accept);

        return Response.ok(OperationResult.OK).build();
    }


    private String getUsername(){
        return securityContext.getUserPrincipal().getName();
    }

}
