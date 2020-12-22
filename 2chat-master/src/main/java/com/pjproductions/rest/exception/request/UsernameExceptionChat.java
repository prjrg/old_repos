package com.pjproductions.rest.exception.request;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.ChatException;

public class UsernameExceptionChat extends ChatException {

    public UsernameExceptionChat(OperationResult codes) {
        super(codes);
    }
}
