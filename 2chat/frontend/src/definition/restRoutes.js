export const REST_API_ROUTE = '/api';

function makeURL(link){
    return REST_API_ROUTE + link;
}

function makeURLParams(...params){
    return REST_API_ROUTE + params.join("/");
}

export const REST_API_LOGIN =  makeURL("/auth/login");
export const REST_API_REGISTER =  makeURL("/auth/register");
const REST_API_USERS = "/user";
export const REST_API_USERS_FRIENDS_REQUESTS = makeURL(REST_API_USERS + "/requests");
export const REST_ACCEPT_REQUEST = (username, accept) => makeURLParams(REST_API_USERS, username, 'request', accept);