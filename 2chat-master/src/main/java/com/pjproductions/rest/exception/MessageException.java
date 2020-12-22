package com.pjproductions.rest.exception;

import com.pjproductions.rest.definition.OperationResult;

public class MessageException extends ChatException{

    public MessageException(OperationResult codes) {
        super(codes);
    }
}
