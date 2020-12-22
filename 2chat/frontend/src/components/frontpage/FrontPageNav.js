import React, {Component} from 'react'
import {
    Nav, Navbar, NavbarBrand, NavbarToggler, Collapse, NavItem, NavLink, Container, Row, Col
} from 'reactstrap'
import '../../css/Styles.css'
import {Link, withRouter} from "react-router-dom";
import Login from "./Login";
import {connect} from "react-redux";

const elemStyle = {
    'paddingTop': '1%',
    'alignContent': 'top'
};


class FrontPageNav extends Component {
    constructor(props) {
        super(props);

        this.toggle = this.toggle.bind(this);
        this.state = {
            isOpen: false
        };
    }

    toggle() {
        this.setState({
            isOpen: !this.state.isOpen
        });
    }

    render(){

        return (
            <div>
            <Navbar className="navbar-dark bg-dark" expand="md">
                <NavbarBrand style={elemStyle} className="py-0" id="logonav" tag={Link} to="/">2Chat</NavbarBrand>
        <NavbarToggler onClick={this.toggle} />
        <Collapse isOpen={this.state.isOpen} navbar>
            <Nav className="ml-auto" navbar>
                <NavItem style={elemStyle}>
                    <Container>
                        <Row>
                            <Col>
                                <Login/>
                            </Col>
                        </Row>
                        <Row>
                            <Col>
                                <NavLink className="label-chat" tag={Link} to="/signup">No account yet? Sign up now!</NavLink>
                            </Col>
                        </Row>
                    </Container>
                </NavItem>
            </Nav>
    </Collapse>
    </Navbar>
    </div>

        )
    }
}



export default withRouter(connect()(FrontPageNav));