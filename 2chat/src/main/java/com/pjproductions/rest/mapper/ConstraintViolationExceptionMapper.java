package com.pjproductions.rest.mapper;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.security.validation.PreCheckImpl;

import javax.validation.ConstraintViolation;
import javax.validation.ConstraintViolationException;
import javax.ws.rs.core.Response;
import javax.ws.rs.ext.ExceptionMapper;
import javax.ws.rs.ext.Provider;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

@Provider
public class ConstraintViolationExceptionMapper implements ExceptionMapper<ConstraintViolationException> {
    @Override
    public Response toResponse(ConstraintViolationException ex) {
        Set<ConstraintViolation<?>> a = ex.getConstraintViolations();
        Optional<ConstraintViolation<?>> b = a.stream().findFirst();

        return b.map(e -> {
            if(new HashSet<>(e.getConstraintDescriptor().getConstraintValidatorClasses()).contains(PreCheckImpl.class)){
                return Response.ok(e.getMessageTemplate()).build();
            }
            return Response.status(Response.Status.BAD_REQUEST).entity(OperationResult.UNACCEPTABLE_PARAMS).build();
        }).orElse(Response.status(Response.Status.BAD_REQUEST).entity(OperationResult.UNACCEPTABLE_PARAMS).build());
    }
}
