package com.pjproductions.rest.exception.request;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.ChatException;

public class EmailExceptionChat extends ChatException {


    public EmailExceptionChat(OperationResult codes) {
        super(codes);
    }
}
