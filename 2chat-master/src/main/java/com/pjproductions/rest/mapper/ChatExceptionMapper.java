package com.pjproductions.rest.mapper;

import com.pjproductions.rest.exception.ChatException;

import javax.ws.rs.core.Response;
import javax.ws.rs.ext.ExceptionMapper;
import javax.ws.rs.ext.Provider;

@Provider
public class ChatExceptionMapper implements ExceptionMapper<ChatException> {
    @Override
    public Response toResponse(ChatException exception) {
        return Response.ok(exception).build();
    }
}
