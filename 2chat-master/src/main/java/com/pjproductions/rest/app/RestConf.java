package com.pjproductions.rest.app;

import org.glassfish.jersey.server.ResourceConfig;
import org.glassfish.jersey.server.filter.RolesAllowedDynamicFeature;

public class RestConf extends ResourceConfig {
    public RestConf() {


        packages(true, "com.pjproductions.rest, com.pjproductions.rest.mapper," +
                "com.fasterxml.jackson.jaxrs.json," +
                "com.pjproductions.rest.security.filter");
        // Define the package which contains the service classes.
        register(RolesAllowedDynamicFeature.class);
        register(com.fasterxml.jackson.jaxrs.json.JacksonJaxbJsonProvider.class);
        register(new AppBinder());
    }
}
