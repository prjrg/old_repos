package com.pjproductions.rest.mapper;

import com.pjproductions.rest.definition.OperationResult;

import javax.ws.rs.core.Response;
import javax.ws.rs.ext.ExceptionMapper;
import javax.ws.rs.ext.Provider;

@Provider
public class RunTimeExceptionMapper implements ExceptionMapper<RuntimeException>{

    @Override
    public Response toResponse(RuntimeException exc) {
        return Response.status(Response.Status.BAD_REQUEST).entity(OperationResult.INVALID_REQUEST).build();
    }
}
