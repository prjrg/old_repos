import React, {Component} from 'react';
import PropTypes from 'prop-types';
import {Button, Card, CardTitle, DropdownItem} from "reactstrap";
import {requestAccepted} from "../../state/ActionsFriends";
import {connect} from "react-redux";

class FriendRequest extends Component {
    constructor(props){
        super(props);
        this.state = {
            username: this.props.username
        };

        this.onSelect = this.onSelect.bind(this);
    }

    onSelect(accept){
        this.props.acceptRequest(this.state.username, accept);
    }

    render() {
        let {username} = this.state;
        return (<DropdownItem>
            <Card body inverse style={{backgroundColor: '#333', borderColor: '#333'}}>
                <CardTitle>{username}</CardTitle>
                <div>
                    <Button onClick={() => this.onSelect(true)}>Add</Button>{' '}
                    <Button onClick={() => this.onSelect(false)}>Decline</Button>
                </div>
            </Card>
        </DropdownItem>);
    }
}

FriendRequest.propTypes = {
    acceptRequest: PropTypes.func.isRequired,
    username: PropTypes.string.isRequired,
};

const mapDispatchToProps = dispatch => {
    return {
        acceptRequest: (username, accept) => (dispatch(requestAccepted(username, accept))),
    }
};

export default connect(null, mapDispatchToProps)(FriendRequest);