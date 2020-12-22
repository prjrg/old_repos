package com.pjproductions.rest.exception.request;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.ChatException;

public class TokenExceptionChat extends ChatException {
    public TokenExceptionChat(OperationResult codes) {
        super(codes);
    }
}
