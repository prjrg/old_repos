package com.pjproductions.rest.mapper;

import com.pjproductions.rest.definition.OperationResult;

import javax.validation.ValidationException;
import javax.ws.rs.core.Response;
import javax.ws.rs.ext.ExceptionMapper;
import javax.ws.rs.ext.Provider;

@Provider
public class ValidationExceptionMapper implements ExceptionMapper<ValidationException>{

    @Override
    public Response toResponse(ValidationException exc) {

        return Response.status(Response.Status.BAD_REQUEST).entity(OperationResult.INVALID_REQUEST).build();
    }
}
