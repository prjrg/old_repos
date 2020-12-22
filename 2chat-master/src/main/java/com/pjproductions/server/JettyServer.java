package com.pjproductions.server;

import com.pjproductions.rest.app.RestConf;
import com.pjproductions.rest.security.PagesErrorHandler;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.handler.HandlerCollection;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.webapp.WebAppContext;
import org.glassfish.jersey.servlet.ServletContainer;

import javax.naming.NamingException;

public class JettyServer {


    public static void main( String[] args ) throws Exception
    {

        Server server = configureServer();

        server.start();

        server.join();

    }

    private static Server configureServer() throws NamingException {

        Server server = new Server(8080);
        PagesErrorHandler errorHandler = new PagesErrorHandler();
        errorHandler.setServer(server);
        errorHandler.setShowStacks(false);

        final HandlerCollection handlers = new HandlerCollection();

        //WebApp
        String resourcesLocation = "src/main/webapp/";
        WebAppContext webAppCtx = new WebAppContext();
        webAppCtx.setContextPath("/");
        webAppCtx.setResourceBase(resourcesLocation);
        webAppCtx.setDescriptor(resourcesLocation + "WEB-INF/web.xml");
        webAppCtx.setErrorHandler(errorHandler);

        handlers.addHandler(webAppCtx);

        //Rest API

        ServletContextHandler servletContext = new ServletContextHandler(ServletContextHandler.NO_SESSIONS);
        servletContext.setContextPath("/api");

        ServletContainer jerseyContainer = new ServletContainer(new RestConf());
        ServletHolder jerseyServletHolder = new ServletHolder(jerseyContainer);
        jerseyServletHolder.setInitOrder(0);

        servletContext.addServlet(jerseyServletHolder, "/*");

        handlers.addHandler(servletContext);

        server.setHandler(handlers);
        server.addBean(errorHandler);

        return server;
    }

}
