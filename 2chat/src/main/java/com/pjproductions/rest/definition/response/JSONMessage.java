package com.pjproductions.rest.definition.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.pjproductions.rest.definition.OperationResult;

import javax.validation.constraints.NotNull;
import java.util.Collections;

@JsonPropertyOrder({"code", "message", "data"})
public class JSONMessage<T>  {

    @JsonProperty("code")
    private final int code;
    @JsonProperty("message")
    private final String message;
    @JsonProperty("data")
    private final T data;

    public JSONMessage(int code, String message, T data) {
        this.code = code;
        this.message = message;
        this.data = data;
    }

    public static <T> JSONMessage<T> of(OperationResult op,@NotNull T data){
        return new JSONMessage<>(op.value(), op.message(), data);
    }

    public static JSONMessage<?> of(OperationResult op){
        return new JSONMessage<>(op.value(), op.message(), Collections.EMPTY_LIST);
    }


    public int getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }

    public T getData() {
        return data;
    }
}
