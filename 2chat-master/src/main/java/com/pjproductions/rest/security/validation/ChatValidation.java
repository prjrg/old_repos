package com.pjproductions.rest.security.validation;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.MessageException;

public class ChatValidation {

    public static String validateMessage(String message) throws MessageException {
        if(message.isEmpty()) throw new MessageException(OperationResult.INVALID_MESSAGE);

        return message;
    }
}
