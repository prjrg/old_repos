package com.pjproductions.rest.definition;

import com.fasterxml.jackson.annotation.JsonValue;
import com.pjproductions.rest.definition.response.JSONMessage;

public enum OperationResult {
    OK("Success"),
    OK_CREATED("Created"),
    OK_ADDED("Added"),
    OK_SENT("Sent"),
    WRONG_USERNAME("Non-existing username"),

    WRONG_PASSWORD("Wrong password"),

    INVALID_USERNAME("Invalid username"),
    INVALID_PASSWORD("Non-acceptable password"),
    INVALID_EMAIL("Invalid email address"),
    NOT_ACCESSABLE("The page you requested is not found"),
    INVALID_OPERATION("Forbidden Operation!"),
    INVALID_CREDENTIALS("Invalid Credentials"),
    INVALID_MESSAGE("Non acceptable message!"),
    INVALID_REQUEST("Invalid request"),

    ILLEGAL_ACTION("Illegal action - user blocked or is already friend"),

    NON_MATCHING_PASSWORD_CONFIRMATION("Password confirmation doesn't match"),
    EXISTING_ACCOUNT("Email already assigned"),
    EXISTING_USERNAME("Existing username account"),
    UNACCEPTABLE_PARAMS("Unacceptable request parameters"),

    NON_EXISTING_CONTENT("Non Existent Content!"),
    NON_EXISTING_ACCOUNT("Non-existing account"),
    NON_EXISTING_CHANNEL("Non existing channel!!"),
    NON_EXISTING_RECEIVER("Non Existent Receiver!"),

    NOT_ASSIGNED("Not assigned!"),
    CREDENTIALS_REQUIRED("REQUIRED CREDENTIALS!"),
    UNAUTHORIZED_ACCESS("Illegal or Unauthorized access"),

    DUPLICATED_OPERATION("Another operation was already submitted, possibly duplicated!");

    private final int code;

    private final String message;

    OperationResult(String message) {
        this.message = message;
        code = this.ordinal();
    }

    public int value() {
        return code;
    }

    public String message(){
        return message;
    }

    @JsonValue
    public JSONMessage<?> getJSONMessage(){
        return JSONMessage.of(this);
    }

}
