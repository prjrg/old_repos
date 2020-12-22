package com.pjproductions.rest.mapper;

import com.pjproductions.rest.exception.PersistenceException;

import javax.ws.rs.core.Response;
import javax.ws.rs.ext.ExceptionMapper;
import javax.ws.rs.ext.Provider;

@Provider
public class PersistenceExceptionMapper implements ExceptionMapper<PersistenceException>{
    @Override
    public Response toResponse(PersistenceException exception) {
        return Response.ok(exception).build();
    }

}
