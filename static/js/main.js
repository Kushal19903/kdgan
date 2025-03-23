document.addEventListener('DOMContentLoaded', function() {
    // Form submission handling
    const generateForm = document.getElementById('generate-form');
    if (generateForm) {
        generateForm.addEventListener('submit', function() {
            const textInput = document.getElementById('text-input');
            if (!textInput.value.trim()) {
                alert('Please enter a text description');
                return false;
            }
            
            const generateBtn = document.getElementById('generate-btn');
            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
            
            return true;
        });
    }
    
    // Close alert buttons
    const closeButtons = document.querySelectorAll('.alert .close');
    closeButtons.forEach(button => {
        button.addEventListener('click', function() {
            this.parentElement.style.display = 'none';
        });
    });
});