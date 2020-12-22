package com.pjproductions.rest.exception.request;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.ChatException;

public class AuthExceptionChat extends ChatException {

    public AuthExceptionChat(OperationResult codes) {
        super(codes);
    }
}
