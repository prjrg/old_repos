package com.pjproductions.rest.exception;

import com.pjproductions.rest.definition.OperationResult;

public class PersistenceException extends ChatException{

    public PersistenceException(OperationResult codes) {
        super(codes);
    }
}
