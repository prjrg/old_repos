package com.pjproductions.providers;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;

public class TimeProvider {

    public TimeProvider() {
    }


    public ZonedDateTime utc(){
        return ZonedDateTime.now(ZoneOffset.UTC);
    }
}
