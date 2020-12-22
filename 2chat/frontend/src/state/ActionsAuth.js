import {REST_API_LOGIN, REST_API_REGISTER} from "../definition/restRoutes";
import {REST_REQUEST} from "../definition/helpers";

export const SET_ROUTING = 'ROUTING';

export const LOGIN_ATTEMPT = 'LOGIN_ATTEMPT';
export const LOGIN_SUCCESSFUL = 'LOGIN_SUCCESSFUL';
export const LOGIN_FAILED = 'LOGIN_FAILED';
export const LOGIN_CLEAR = 'LOGIN_CLEAR';

export const REGISTER_ATTEMPT = 'REGISTER_ATTEMPT';
export const REGISTER_SUCCESSFUL = 'REGISTER_SUCCESSFULL';
export const REGISTER_FAILED = 'REGISTER_FAILED';
export const REGISTER_CLEAR = 'REGISTER_CLEAR';

export const SET_USERNAME = 'SET_USERNAME';
export const CLEAR_USERNAME = 'CLEAR_USERNAME';
export const AUTHENTICATED = 'USER_AUTHENTICATED';
export const RECEIVE_TOKEN = 'RECEIVE_TOKEN';

const jwtStore = "C5431AZPWQz32";
const setAuth = (token) => (localStorage.setItem(jwtStore, token));
export const authToken = () => (localStorage.getItem(jwtStore));


export function makeActionCreator(type, ...argNames) {
    return function (...args) {
        let action = { type };
        argNames.forEach((arg, index) => {
            action[argNames[index]] = args[index]
        });
        return action
    }
}

export const routing = makeActionCreator(SET_ROUTING, 'route', 'username');

export function login(username, password){

    return function(dispatch){
        dispatch(loginAttempt(username));

        return fetch(REST_API_LOGIN, REST_REQUEST.loginRequest(username, password))
            .then(response => response.json())
            .then(json => {
                if (json.code === 0) {
                    dispatch(loginSuccess(username));
                    setAuth(json.data);
                }
                else {
                    dispatch(loginFailed(json.code, json.message))
                }
            })
            .catch((error) => console.error(error));
    }
}

export function signUp(username, email, password, password2){

    return function(dispatch){
        dispatch(registerAttempt());

        return fetch(REST_API_REGISTER, REST_REQUEST.registerRequest(username, email, password, password2))
            .then(response =>  response.json())
            .then(json => {
                if (json.code === 1) {
                    dispatch(registerSuccess(username));
                }
                else {
                    dispatch(registerFailed(json.code, json.message))
                }
            })
            .catch((error) => {console.log(error)});
    }
}

export const loginAttempt = makeActionCreator(LOGIN_ATTEMPT, 'username');
export const loginFailed = makeActionCreator(LOGIN_FAILED, 'code', 'error');
export const loginSuccess = makeActionCreator(LOGIN_SUCCESSFUL, 'username');

export const registerAttempt = makeActionCreator(REGISTER_ATTEMPT);
export const registerFailed = makeActionCreator(REGISTER_FAILED, 'code', 'error');
export const registerSuccess = makeActionCreator(REGISTER_SUCCESSFUL, 'username');
export const defaultRegister = makeActionCreator(REGISTER_CLEAR);

export const setUsername = makeActionCreator(SET_USERNAME, 'username');
export const clearUsername = makeActionCreator(CLEAR_USERNAME);
const authenticateActionHelper = makeActionCreator(AUTHENTICATED, 'status');
export const notAuthenticated = () => authenticateActionHelper(false);
export const authenticated = () => authenticateActionHelper(true);

export function receiveToken(json){
    return {
        type: RECEIVE_TOKEN,
        token: json.data
    }
}



