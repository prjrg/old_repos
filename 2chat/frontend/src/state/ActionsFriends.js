import {makeActionCreator} from "./ActionsAuth";
import {REST_ACCEPT_REQUEST, REST_API_USERS, REST_API_USERS_FRIENDS_REQUESTS} from "../definition/restRoutes";
import {REST_REQUEST} from "../definition/helpers";

export const FRIEND_REQUESTS = "friend_requests";
export const LOADING_FRIEND_REQUESTS = "loading_requests";
export const FRIEND_REQUESTS_SUCCESSFUL = "all_friend_requests";
export const FRIEND_REQUESTS_FAILED = "failed_friend_requests";

export const askFriendRequests = makeActionCreator(FRIEND_REQUESTS);
export const loadingFriendRequests = makeActionCreator(LOADING_FRIEND_REQUESTS);
export const friendsRequestSuccess = makeActionCreator(FRIEND_REQUESTS_SUCCESSFUL, 'requests');
export const failedProcess = makeActionCreator(FRIEND_REQUESTS_FAILED, 'code', 'error');


export const fetchFriendRequests = () => {

    return (dispatch) => {
        dispatch(loadingFriendRequests);

        return fetch(REST_API_USERS_FRIENDS_REQUESTS, REST_REQUEST.makeTokenRequest('GET'))
            .then(response => response.json())
            .then(json => {
                if (json.code === 0) {
                    dispatch(friendsRequestSuccess(json.data));
                }
                else {
                    dispatch(failedProcess(json.code, json.message))
                }
            })
            .catch((error) => console.error(error));
    }
};

export const requestAccepted = (username, accept) => {
    return (dispatch) => {

        return fetch(REST_ACCEPT_REQUEST(username, accept), REST_REQUEST.makeTokenRequest('GET'))
            .then(response => response.json())
            .then(json => {
                if(json.code === 0){
                    dispatch(fetchFriendRequests());
                }
            })
    }
};

