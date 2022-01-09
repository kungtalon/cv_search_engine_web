import SearchBox from "./SearchBox"
import { Navbar, Container, Nav } from "react-bootstrap"

function Header(props) {
  return (
    <div>
      <Navbar bg="dark" variant="dark" sticky="top">
        <Container>
        <Navbar.Brand>CV Search</Navbar.Brand>
        <Nav className="me-auto">
          <Nav.Link href="" onClick={props.onClickHome}>Home</Nav.Link>
          <Nav.Link href="" onClick={props.onClickIntro}>Intro</Nav.Link>
          {/* <Nav.Link href="#feedback">Feedback</Nav.Link> */}
          <Nav.Link href="https://github.com/kungtalon"><i className="icon fa fa-github"></i>Github</Nav.Link>
        </Nav>

        <SearchBox onSearch={props.onSearch}/>
        </Container>
      </Navbar>
      <br />
    </div>
  );
}

export default Header;
