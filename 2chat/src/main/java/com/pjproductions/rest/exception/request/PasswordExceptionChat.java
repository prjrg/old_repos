package com.pjproductions.rest.exception.request;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.ChatException;

public class PasswordExceptionChat extends ChatException {

    public PasswordExceptionChat(OperationResult codes) {
        super(codes);
    }
}
