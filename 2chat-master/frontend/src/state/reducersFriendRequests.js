

import {FRIEND_REQUESTS_FAILED, FRIEND_REQUESTS_SUCCESSFUL, LOADING_FRIEND_REQUESTS} from "./ActionsFriends";
import {handleActions} from "redux-actions";

const initialState = {
    friendRequests: {
        fetch: {isFetching: false, code: 0, error: "", fetched: false},
        requests: []
    },
};

const fRequestsReducers = {
    [LOADING_FRIEND_REQUESTS]: (state, action) => ({
        ...state,
        fetch: {...fetch, isFetching: true}
    }),
    [FRIEND_REQUESTS_SUCCESSFUL]: (state, action) => ({
        ...state,
        fetch: {isFetching: false, code: 0, error: "", fetched: true},
        requests: action.requests
    }),

    [FRIEND_REQUESTS_FAILED]: (state, action) => ({
        ...state,
        fetch: {isFetching: false, code: action.code, error: action.error, fetched: false}
    })
};

const friendRequests = handleActions(fRequestsReducers, initialState.friendRequests);

export default friendRequests;