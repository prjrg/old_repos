package com.pjproductions.rest.exception.request;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.ChatException;

public class AccountExceptionChat extends ChatException {

    public AccountExceptionChat(OperationResult codes) {
        super(codes);
    }
}
