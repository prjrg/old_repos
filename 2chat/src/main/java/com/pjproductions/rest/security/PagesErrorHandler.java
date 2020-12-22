package com.pjproductions.rest.security;

import org.eclipse.jetty.server.handler.ErrorHandler;

import javax.servlet.http.HttpServletRequest;
import java.io.IOException;
import java.io.Writer;

public class PagesErrorHandler extends ErrorHandler {

    @Override
    protected void writeErrorPage(HttpServletRequest request, Writer writer, int code, String message, boolean showStacks) throws IOException {
                writer.append("<html>\n<head>\n <meta http-equiv=\"Content-Type\" content=\"text/html;charset=utf-8\"/>\n")
                .append("<title>Error ")
                .append(String.valueOf(code))
                .append(" ")
                .append(message)
                .append("</title>\n</head>\n")
                .append("<body>\n<h2>HTTP ERROR ")
                .append(String.valueOf(code))
                .append("</h2>\n<p>Problem accessing ")
                .append(request.getRequestURI())
                .append("Reason:\n\n")
                .append("<pre> ")
                .append(message)
                .append(" </pre>\n</p>\n<hr>\n<a href=\"http://tochat.com\">")
                .append("Powered by PJProductions</a>\n<hr/>\n</body>\n</html>");
    }

}
