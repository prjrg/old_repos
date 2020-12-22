package com.pjproductions.persistence.controllers;

import com.pjproductions.rest.definition.json.MessageJSON;
import com.pjproductions.rest.definition.json.ReadMessages;
import com.pjproductions.rest.definition.json.ReadNewMessages;
import com.pjproductions.rest.exception.ChatException;

import java.util.Collection;
import java.util.Map;


public interface MessengerDataController {


    <U extends ChatException> void sendOnePrivate(String from, String to, String message) throws U;
    <U extends ChatException> void sendMultiplePrivate(String from, Collection<String> to, String message) throws U;


    <U extends ChatException> Map<String, Collection<MessageJSON<String>>> readManyUsers(String username, Collection<ReadMessages<String>> friends) throws U;
    <U extends ChatException> Collection<MessageJSON<String>> readOneUser(String u, ReadMessages<String> friend) throws U;
    <U extends ChatException> Map<String, Collection<MessageJSON<String>>> readAll(String username) throws U;

    <U extends ChatException> Map<String, Collection<MessageJSON<String>>> readNewManyUsers(String username, Collection<ReadNewMessages<String>> friends) throws U;
    <U extends ChatException> Collection<MessageJSON<String>> readNewOneUser(String u, ReadNewMessages<String> friend) throws U;

    <U extends ChatException> long lastIndex(String username, String friend) throws U;
    <U extends ChatException> Map<String, Long> lastIndex(String username, Collection<String> friend) throws U;
    <U extends ChatException> Map<String, Long> lastIndex(String username) throws U;


}
