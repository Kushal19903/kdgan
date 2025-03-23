import React from 'react';
import { Link, NavLink } from 'react-router-dom';

function Header() {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <Link to="/" className="logo">KD-GAN</Link>
          <nav>
            <ul className="nav-links">
              <li>
                <NavLink 
                  to="/" 
                  className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
                  end
                >
                  Home
                </NavLink>
              </li>
              <li>
                <NavLink 
                  to="/generate" 
                  className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
                >
                  Generate
                </NavLink>
              </li>
              <li>
                <NavLink 
                  to="/gallery" 
                  className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
                >
                  Gallery
                </NavLink>
              </li>
              <li>
                <NavLink 
                  to="/about" 
                  className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
                >
                  About
                </NavLink>
              </li>
            </ul>
          </nav>
        </div>
      </div>
    </header>
  );
}

export default Header;