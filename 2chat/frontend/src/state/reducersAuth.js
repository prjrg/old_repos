import {
    LOGIN_ATTEMPT, LOGIN_FAILED, LOGIN_SUCCESSFUL, REGISTER_FAILED, REGISTER_ATTEMPT, REGISTER_SUCCESSFUL, REGISTER_CLEAR,
} from "./ActionsAuth";
import {handleActions} from "redux-actions";

const initialState = {
        username: "",
        login: {loggingIn: false, loginError: 0, loggedIn: false},
        register: {registering: false, registerError: 1, registered: false},
        authenticated: false,
};

const authReducers = {
    [REGISTER_CLEAR]: (state, action) => ({
        ...state,
        register: initialState.register
    }),
    [REGISTER_SUCCESSFUL]: (state, action) => ({
        ...state,
        register: {registering: false, registerError: 1, registered: true},
        username: action.username
    }),

    [REGISTER_FAILED]: (state, action) => ({
        ...state,
        register: {...state.register, registering: false, registerError: action.code},
    }),

    [REGISTER_ATTEMPT]: (state, action) => ({
        ...state,
        register: {...state.register, registering: true, registerError: 1},
        username: ""
    }),
    [LOGIN_SUCCESSFUL]: (state, action) => ({
        ...state,
        login: {loggingIn: false, loginError: 0, loggedIn: true},
        username: action.username,
        authenticated: true
    }),

    [LOGIN_FAILED]: (state, action) => ({
        ...state,
        login: {loggingIn: false, loginError: action.code},
    }),

    [LOGIN_ATTEMPT]: (state, action) => ({
        ...state,
        login: {loggingIn: true, loginError: 0},
        username: action.username,
        authenticated: false
    })
};

const authentication = handleActions(authReducers, initialState);



export default authentication;