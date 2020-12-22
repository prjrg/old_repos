package com.pjproductions.rest.security.validation;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.ChatException;
import com.pjproductions.rest.exception.request.PasswordExceptionChat;
import com.pjproductions.rest.security.validation.validators.CheckValid;
import com.pjproductions.rest.security.validation.validators.CheckValidators;

import javax.servlet.http.HttpServletResponse;
import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;
import javax.validation.constraintvalidation.SupportedValidationTarget;
import javax.validation.constraintvalidation.ValidationTarget;
import javax.ws.rs.core.Context;
import java.lang.annotation.AnnotationFormatError;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

@SupportedValidationTarget(ValidationTarget.PARAMETERS)
public class PreCheckImpl implements ConstraintValidator<PreCheck, Object[]>{
    private static final Map<String, CheckValid> validators;
    @Context
    private HttpServletResponse response;

    static {
        validators = new ConcurrentHashMap<>();
        PreCheckImpl.registerValidator("username", new CheckValidators.CheckUsername());
        PreCheckImpl.registerValidator("password", new CheckValidators.CheckPassword());
        PreCheckImpl.registerValidator("email", new CheckValidators.CheckEmail());
    }

    public static void registerValidator(String name, CheckValid validator){
        validators.put(name, validator);
    }

    private PreCheck check;


    @Override
    public void initialize(PreCheck preCheck) {
        this.check = preCheck;
    }

    @Override
    public boolean isValid(Object[] params, ConstraintValidatorContext constraintValidatorContext) {

        String[] validationType = check.params();

        if(params == null) throw new AnnotationFormatError("@PreCheck-null");

        String password1 = null;
        String password2 = null;
        for(int i=0;i<validationType.length;++i){

            if(params[i] instanceof String) {
                String param = (String) params[i];
                if(validationType[i].equals("password")) password1 = param;

                if(validationType[i].equals("password2")) password2 = param;
                else {
                    final Optional<ChatException> invalid = validators.get(validationType[i]).isValid(param);
                    if (invalid.isPresent()) {
                        constraintValidatorContext.disableDefaultConstraintViolation();
                        constraintValidatorContext
                                .buildConstraintViolationWithTemplate(invalid.get().asJSONString())
                                .addConstraintViolation();
                        return false;

                    }
                }
            }
        }

        if(password1 != null && password2 != null && !password1.equals(password2)){
            constraintValidatorContext.disableDefaultConstraintViolation();
            constraintValidatorContext
                    .buildConstraintViolationWithTemplate(new PasswordExceptionChat(OperationResult.NON_MATCHING_PASSWORD_CONFIRMATION).asJSONString())
                    .addConstraintViolation();
            return false;
        }

        return true;
    }

}
