package com.pjproductions.rest.exception;


import com.fasterxml.jackson.annotation.JsonValue;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.definition.response.JSONMessage;


public class ChatException extends RuntimeException{
    private OperationResult response;

    public ChatException(OperationResult codes) {
        super("");
        this.response = codes;
    }

    @JsonValue
    public JSONMessage<?> result() {
        return JSONMessage.of(response);
    }

    public String asJSONString(){
        try {
            return new ObjectMapper().writeValueAsString(response.getJSONMessage());
        } catch (JsonProcessingException e) {
            return "";
        }
    }
}
