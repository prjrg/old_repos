import React, {Component} from 'react'
import {
    Nav, Navbar, NavbarBrand, NavbarToggler, Dropdown, DropdownToggle, Collapse, DropdownItem, DropdownMenu
} from 'reactstrap'
import '../../css/Styles.css'
import {Link} from "react-router-dom";
import FontAwesome from 'react-fontawesome';
import {connect} from "react-redux";
import PropTypes from 'prop-types';
import {fetchFriendRequests} from "../../state/ActionsFriends";
import Requests from "./Requests";



const elemStyle = {
    'paddingTop': '1%',
    'alignContent': 'top'
};


class HomeNav extends Component {
    constructor(props) {
        super(props);

        this.toggleNewfriend = this.toggleNewfriend.bind(this);
        this.toggle = this.toggle.bind(this);
        this.toggleNotifications = this.toggleNotifications.bind(this);
        this.state = {
            isOpen: false,
            addFriendOpen: false,
            notificationsOpen: false,
            requests: []
        };
    }

    toggleNewfriend() {
        this.setState({
            addFriendOpen: !this.state.addFriendOpen
        });
    }

    toggleNotifications() {
        if(!this.state.notificationsOpen) this.props.fetchRequests();
        this.setState({
            notificationsOpen: !this.state.notificationsOpen
        });
    }

    toggle(){
        this.setState({
            isOpen: !this.state.isOpen
        })
    }

    render(){

        const {requests} = this.state;

        return (
            <div>
                <Navbar className="navbar-dark bg-dark" expand="sm">
                    <NavbarBrand style={elemStyle} id="logonav" tag={Link} to="/">2Chat</NavbarBrand>
                    <NavbarToggler onClick={this.toggle} />
                    <Collapse isOpen={this.state.isOpen} navbar>
                        <Nav className="ml-lg-auto" tabs>
                            <Dropdown nav isOpen={this.state.addFriendOpen} toggle={this.toggleNewfriend}>
                                <DropdownToggle nav>
                                    <span className="nav-items-chat">
                                    <FontAwesome
                                        name='user-plus'
                                        size='lg'
                                        ariaLabel="Add Friend"
                                    />
                                </span>
                                </DropdownToggle>
                            </Dropdown>
                            <Dropdown nav isOpen={this.state.notificationsOpen} toggle={this.toggleNotifications}>
                                <DropdownToggle nav>
                                    <span className="nav-items-chat">
                                    <FontAwesome
                                        name='bell-o'
                                        size='lg'
                                        ariaLabel="Notifications"
                                    />
                                    </span>
                                </DropdownToggle>
                                <DropdownMenu>
                                    <Requests/>
                            </DropdownMenu>
                            </Dropdown>
                        </Nav>
                    </Collapse>
                </Navbar>
            </div>

        )
    }
}

HomeNav.propTypes = {
    fetchRequests: PropTypes.func.isRequired,
};

const mapStateToProps = (state) => ({});

const mapDispatchToProps = dispatch => {
    return {
        fetchRequests: () => (dispatch(fetchFriendRequests())),
    }
};

export default connect(mapStateToProps, mapDispatchToProps)(HomeNav);