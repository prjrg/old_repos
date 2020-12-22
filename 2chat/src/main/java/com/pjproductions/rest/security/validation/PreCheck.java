package com.pjproductions.rest.security.validation;

import javax.validation.Constraint;
import javax.validation.ConstraintTarget;
import javax.validation.Payload;
import java.lang.annotation.*;

@Target({ElementType.PARAMETER, ElementType.FIELD, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = PreCheckImpl.class)
@Documented
public @interface PreCheck {

    String[] params() default {};

    String message() default "PreCheckValidation";
    Class<?>[] groups() default {};
    Class<? extends Payload>[] payload() default {};
}
