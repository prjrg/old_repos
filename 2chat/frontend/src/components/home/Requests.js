import React, {Component} from 'react';
import PropTypes from 'prop-types';
import {Alert, DropdownMenu} from "reactstrap";
import FriendRequest from "./FriendRequest";
import {connect} from "react-redux";

class Requests extends Component {
    constructor(props){
        super(props);
        this.state = {
            requests: this.props.requests,
        }
    }

    componentWillReceiveProps(nextProps){
        if(nextProps !== this.props){
            this.setState((state) => ({...state, requests: nextProps.requests}));
        }
    }

    render(){

        const {requests} = this.state;

        return (requests.length > 0) ?
            <div>
                {requests.map(reqToItem)}
            </div>
            :
            <div>
                <Alert color="dark">
                    <h4 className="alert-heading">No New Friend Requests</h4>
                    <hr/>
                    <p className="mb-0">
                        Psst!
                    </p>
                </Alert>
            </div>
        }
}

const reqToItem = (request) => (<FriendRequest username={request}/>);

Requests.propTypes = {
    requests: PropTypes.object.isRequired,
};

const mapStateToProps = (state) => {
    const {friendRequests} = state;
    const {requests} = friendRequests;
    return {
        requests,
    };
};

export default connect(mapStateToProps)(Requests);