package com.pjproductions.persistence.controllers;

import com.pjproductions.persistence.storage.data.*;
import com.pjproductions.persistence.storage.data.tofutureuse.*;
import com.pjproductions.rest.exception.ChatException;

import java.util.Collection;

public interface ChatDataController {

    <U extends ChatException> void createChannel(String username, String channelName) throws U;
    <U extends ChatException> void deleteChannel(String user, String channelName) throws U;
    <U extends ChatException> void assignUserToChannel(String username, String channel) throws U;

    <U extends ChatException> Messages<ChannelMessage> getChannelBoundedMessages(String username, String channel, int first, int last) throws U;
    <U extends ChatException> Messages<ChannelMessage> getChannelMessages(String username, String channel) throws U;

    <U extends ChatException> void sendOne(String from, String to, String message, String channel) throws U;
    <U extends ChatException> void sendMultiple(String from, Collection<String> to, String message, String channel) throws U;
    <U extends ChatException> void sendAll(String from, String message, String channel) throws U;

    <U extends ChatException> Room findChannel(String room) throws U;
    <U extends ChatException> RoomsList channels(String user, String room) throws U;
    <U extends ChatException> ChatDescriptor chatDescription(String user, String room) throws U;
    <U extends ChatException>RoomView roomInformation(String user, String room) throws U;


}
