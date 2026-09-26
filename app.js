
class User {
    login(email, password) {
        let validEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
        if (!validEmail || password.length < 6) {
            console.log('Invalid email or password');
        } 
        else {
            console.log('Login successful');
        }
    signUp(email,)
    }
    
}

