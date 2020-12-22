import {authToken} from "../state/ActionsAuth";
import 'url-search-params-polyfill';

export const REST_REQUEST = {
    loginRequest,
    registerRequest,
    makeTokenRequest
};

function loginRequest(user, password){
    return {
        method: 'POST',
        headers: {
            'accept': 'application/json',
            'content-type': "application/x-www-form-urlencoded",
            'authorization': 'Basic ' + new Buffer(`${user}:${password}`).toString('base64')
        }
    };
}

function registerRequest(user, email, password, password2){
    return {
        method: 'POST',
        headers: {
            'accept': 'application/json',
            'content-type': "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({"username": user, "email": email, "password": password, "password2": password2})
    };
}

function makeTokenRequest(method, args={}){

    return (method !== 'GET') ? {
        method: method,
        headers: {
            'accept': 'application/json',
            'authorization': 'Bearer ' + authToken(),
        },
        body: new URLSearchParams({...args})
    }
    :
        {
            method: method,
            headers: {
                'accept': 'application/json',
                'content-type': "application/x-www-form-urlencoded",
                'authorization': 'Bearer ' + authToken()
            }
        };

}