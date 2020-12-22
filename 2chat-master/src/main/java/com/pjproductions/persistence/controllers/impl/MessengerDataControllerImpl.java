package com.pjproductions.persistence.controllers.impl;

import com.pjproductions.persistence.controllers.MessengerDataController;
import com.pjproductions.persistence.storage.Storage;
import com.pjproductions.persistence.storage.data.Message;
import com.pjproductions.persistence.storage.data.User;
import com.pjproductions.providers.TimeProvider;
import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.definition.json.MessageJSON;
import com.pjproductions.rest.definition.json.ReadMessages;
import com.pjproductions.rest.definition.json.ReadNewMessages;
import com.pjproductions.rest.exception.ChatException;
import com.pjproductions.rest.exception.MessageException;

import java.time.ZonedDateTime;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

import static com.pjproductions.rest.security.validation.ChatValidation.validateMessage;

public class MessengerDataControllerImpl implements MessengerDataController {

    private final TimeProvider utc;
    private final Storage storage;
    private final Function<String, User> findUserByEmailOrName;
    private final Function<Long, User> findUserById;

    public MessengerDataControllerImpl(Storage storage) {
        this.storage = storage;
        utc = new TimeProvider();
        findUserByEmailOrName = storage.USER_OPERATIONS::findUserByEmailOrName;
        findUserById = storage.USER_OPERATIONS::findUserById;
    }

    private <U extends ChatException> void sendAux(User u, User u1, String message, ZonedDateTime utc) throws U{
        if(u1.getId().equals(u.getId())){
            throw new MessageException(OperationResult.INVALID_OPERATION);
        }

        Message msg = new Message(u.getId(), u1.getId(), message,  utc, false);
        storage.MESSAGES_OPERATIONS.sendMessage(u, u1, msg);
    }


    @Override
    public <U extends ChatException> void sendOnePrivate(String from, String to, String message) throws U {
        User u = findUserByEmailOrName.apply(from);
        User u1 = findUserByEmailOrName.apply(to);

        message = validateMessage(message);


        sendAux(u, u1, message, utc.utc());
    }

    @Override
    public <U extends ChatException> void sendMultiplePrivate(String from, Collection<String> to, String message) throws U {
        User u = findUserByEmailOrName.apply(from);

        final String message1 = validateMessage(message);

        to.forEach(ui -> {
            final User u1 = findUserByEmailOrName.apply(ui);
            sendAux(u, u1, message1, utc.utc());
        });
    }

    @Override
    public <U extends ChatException> Map<String, Collection<MessageJSON<String>>> readManyUsers(String username, Collection<ReadMessages<String>> friends) throws U {
        User u = findUserByEmailOrName.apply(username);

        Map<String, Collection<MessageJSON<String>>> res = storage.MESSAGES_OPERATIONS
                .manyUsersMessages(u, friends, findUserByEmailOrName, MESSAGE_TO_JSON);

        if(res.size() != friends.size()) throw new ChatException(OperationResult.INVALID_OPERATION);

        return res;
    }

    @Override
    public <U extends ChatException> Collection<MessageJSON<String>> readOneUser(String u, ReadMessages<String> friend) throws U {
        User u1 = findUserByEmailOrName.apply(u);
        User u2 = findUserByEmailOrName.apply(friend.getFriend());

        Collection<MessageJSON<String>> res = storage.MESSAGES_OPERATIONS.oneUserMessages(u1, u2, friend.getFrom(), friend.getTo(), MESSAGE_TO_JSON);

        if(Objects.isNull(res)) throw new ChatException(OperationResult.INVALID_OPERATION);

        return res;
    }

    @Override
    public <U extends ChatException> Map<String, Collection<MessageJSON<String>>> readAll(String username) throws U {
        User u1 = findUserByEmailOrName.apply(username);
        return storage.MESSAGES_OPERATIONS.<MessageJSON<String>, String>allMessages(u1, ID_TO_USERNAME, MESSAGE_TO_JSON);
    }

    @Override
    public <U extends ChatException> Map<String, Collection<MessageJSON<String>>> readNewManyUsers(String username, Collection<ReadNewMessages<String>> friends) throws U {
        User u = findUserByEmailOrName.apply(username);

        Map<String, Collection<MessageJSON<String>>> res = storage.MESSAGES_OPERATIONS
                .manyUsersNewMessages(u, friends, findUserByEmailOrName, MESSAGE_TO_JSON);

        if(res.size() != friends.size()) throw new ChatException(OperationResult.INVALID_OPERATION);

        return res;
    }

    @Override
    public <U extends ChatException> Collection<MessageJSON<String>> readNewOneUser(String u, ReadNewMessages<String> friend) throws U {
        User u1 = findUserByEmailOrName.apply(u);
        User u2 = findUserByEmailOrName.apply(friend.friend());

        Collection<MessageJSON<String>> res = storage.MESSAGES_OPERATIONS.oneUserNewMessages(u1, u2, friend.fromIndex(), MESSAGE_TO_JSON);

        if(res == null) throw new ChatException(OperationResult.INVALID_OPERATION);

        return res;
    }

    @Override
    public <U extends ChatException> long lastIndex(String username, String friend) throws U {
        User u1 = findUserByEmailOrName.apply(username);
        User u2 = findUserByEmailOrName.apply(friend);

        return storage.MESSAGES_OPERATIONS.lastIndex(u1, u2);
    }

    @Override
    public <U extends ChatException> Map<String, Long> lastIndex(String username, Collection<String> friend) throws U {
        User u1 = findUserByEmailOrName.apply(username);

        return storage.MESSAGES_OPERATIONS.lastIndex(u1, friend);
    }

    @Override
    public <U extends ChatException> Map<String, Long> lastIndex(String username) throws U {
        User u1 = findUserByEmailOrName.apply(username);

        return storage.MESSAGES_OPERATIONS.lastIndex(u1);
    }

    private final BiFunction<Long, Message, MessageJSON<String>> MESSAGE_TO_JSON = new BiFunction<Long, Message, MessageJSON<String>>() {
        @Override
        public MessageJSON<String> apply(Long aLong, Message message) {
            return new MessageJSON<>(aLong.intValue(),
                    findUserById.apply(message.getSender()).getUsername(),
                    findUserById.apply(message.getReceiver()).getUsername(),
                    message.getMessage(), message.getTimestamp());
        }
    };

    private final Function<Long, String> ID_TO_USERNAME = new Function<Long, String>() {
        @Override
        public String apply(Long aLong) {
            return storage.USER_OPERATIONS.findUserById(aLong).getUsername();
        }
    };





}
