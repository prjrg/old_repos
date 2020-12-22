package com.pjproductions.rest.security.validation.validators;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.request.EmailExceptionChat;
import com.pjproductions.rest.exception.request.PasswordExceptionChat;
import com.pjproductions.rest.exception.request.UsernameExceptionChat;
import org.apache.commons.validator.routines.EmailValidator;

import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CheckValidators {

    public static class CheckUsername implements  CheckValid<String, UsernameExceptionChat> {
        public static final int USERNAME_MAXSIZE = 20;
        public static final int USERNAME_MINSIZE = 6;

        private static final Pattern pattern = Pattern.compile("\\p{Alnum}+");

        @Override
        public Optional<UsernameExceptionChat> isValid(String username){
            Matcher matcher = pattern.matcher(username);

            if(!(matcher.matches() && username.length() <= USERNAME_MAXSIZE && username.length() >= USERNAME_MINSIZE)) {
                return Optional.of(new UsernameExceptionChat(OperationResult.INVALID_USERNAME));
            }

            return Optional.empty();
        }
    }

    public static class CheckPassword implements CheckValid<String, PasswordExceptionChat>{
        private static final Pattern passPattern = Pattern.compile("^(?=.*\\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()_+\\-=\\[\\]{};’:”\\\\|,.<>\\/?]).{8,20}$");

        @Override
        public Optional<PasswordExceptionChat> isValid(String password) {
               if(!passPattern.matcher(password).matches()){
                   return Optional.of(new PasswordExceptionChat(OperationResult.INVALID_PASSWORD));
               }

               return Optional.empty();
        }
    }

    public static class CheckEmail implements CheckValid<String, EmailExceptionChat> {

        @Override
        public Optional<EmailExceptionChat> isValid(String email) {

            if(EmailValidator.getInstance().isValid(email)){
                return Optional.empty();
            }

            return Optional.of(new EmailExceptionChat(OperationResult.INVALID_EMAIL));
        }
    }

}
