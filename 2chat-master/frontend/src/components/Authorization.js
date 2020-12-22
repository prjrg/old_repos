import React, {Component} from 'react';
import {connect} from "react-redux";
import {withRouter} from "react-router-dom";
import PropTypes from 'prop-types';

class Authorization extends Component {
        constructor(props){
            super(props);
            this.state = {
                authenticated: this.props.authenticated
            }
        }

        componentWillReceiveProps(nextProps){
            if(nextProps.authenticated !== this.state.authenticated){
                this.setState((state) => ({...state, authenticated: !state.authenticated}));
            }

        }

        render(){
            const auth = this.state.authenticated;
            const {CompAuth, CompOpt} = this.props;
            return (auth ? <CompAuth/> : <CompOpt/>);
        }
}

Authorization.propTypes = {
    authenticated: PropTypes.bool.isRequired
};

const mapStateToProps = (state) => {
    const {authentication} = state;
    const {authenticated} = authentication;
    return {authenticated};
};

export default withRouter(connect(mapStateToProps)(Authorization));


