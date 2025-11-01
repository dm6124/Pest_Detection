document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileUpload');
    const imagePreview = document.getElementById('imagePreview');
    const fileName = document.getElementById('fileName');
    const previewContainer = document.getElementById('previewContainer');
    const detectButton = document.getElementById('detectButton');
    const removeButton = document.getElementById('removeButton');
    const resultContainer = document.getElementById('resultContainer');
    const resultContent = document.getElementById('resultContent');
    const chatContainer = document.getElementById('chatContainer');
    const chatLog = document.getElementById('chatLog');
    const chatInput = document.getElementById('chatInput');
    const chatSendButton = document.getElementById('chatSendButton');
    const toggleChat = document.getElementById('toggleChat');
    const quickResponses = document.getElementById('quickResponses');
    
    // State variables
    let detectedPest = '';
    let isScrolledToBottom = true;
    let lastScrollHeight = 0;
    let conversationStep = 0;
    let userCrop = '';
    let managementPreference = '';
    
    // Add a scroll-to-bottom button to the chat container
    const scrollToBottomBtn = document.createElement('button');
    scrollToBottomBtn.innerHTML = '↓';
    scrollToBottomBtn.className = 'scroll-to-bottom-btn';
    scrollToBottomBtn.style.display = 'none';
    scrollToBottomBtn.addEventListener('click', scrollToBottom);
    chatContainer.appendChild(scrollToBottomBtn);
    
    // Handle file upload via click
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
    });
    
    // Handle file drop
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length) handleFiles(files);
    }
    
    // Handle file selection via input
    fileInput.addEventListener('change', function() {
        if (this.files.length) handleFiles(this.files);
    });
    
    function handleFiles(files) {
        const file = files[0];
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }
        
        fileName.textContent = file.name;
        const reader = new FileReader();
        
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            detectButton.disabled = false;
            resultContainer.style.display = 'none';
            
            // Match chat container height to upload container
            matchContainerHeights();
        }
        
        reader.readAsDataURL(file);
    }
    
    // Function to match chat container height to upload container
    function matchContainerHeights() {
        const uploadContainer = document.querySelector('.upload-container');
        const chatContainer = document.querySelector('.chat-container');
        const rightPanel = document.querySelector('.right-panel');
        
        if (uploadContainer && chatContainer && rightPanel) {
            // Set chat container height to match right panel
            chatContainer.style.height = rightPanel.clientHeight + 'px';
            
            // Calculate and set chat log height
            const chatHeader = document.querySelector('.chat-header');
            const chatInputContainer = document.querySelector('.chat-input-container');
            const quickResponsesContainer = document.querySelector('.quick-responses');
            
            const headerHeight = chatHeader ? chatHeader.offsetHeight : 0;
            const inputHeight = chatInputContainer ? chatInputContainer.offsetHeight : 0;
            const quickResponsesHeight = quickResponsesContainer ? quickResponsesContainer.offsetHeight : 0;
            
            const chatLog = document.querySelector('.chat-log');
            if (chatLog) {
                const availableHeight = chatContainer.clientHeight - headerHeight - inputHeight - quickResponsesHeight;
                chatLog.style.height = availableHeight + 'px';
                chatLog.style.maxHeight = availableHeight + 'px';
            }
        }
    }
    
    // Remove image
    removeButton.addEventListener('click', function() {
        imagePreview.src = '';
        fileName.textContent = 'No file selected';
        previewContainer.style.display = 'none';
        detectButton.disabled = true;
        fileInput.value = '';
        resultContainer.style.display = 'none';
        chatContainer.style.display = 'none';
        detectedPest = '';
        
        // Reset conversation
        resetConversation();
    });
    
    // Reset conversation state
    function resetConversation() {
        conversationStep = 0;
        userCrop = '';
        managementPreference = '';
        
        // Clear the chat log
        chatLog.innerHTML = '';
        
        // Reset quick responses
        quickResponses.innerHTML = '';
    }
    
    // Enhanced scroll event listener
    chatLog.addEventListener('scroll', function() {
        const scrollPosition = chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight;
        isScrolledToBottom = scrollPosition < 10;
        
        // Show/hide scroll to bottom button
        if (!isScrolledToBottom && chatLog.scrollHeight > chatLog.clientHeight) {
            scrollToBottomBtn.style.display = 'block';
        } else {
            scrollToBottomBtn.style.display = 'none';
        }
    });
    
    // Detect pests
    detectButton.addEventListener('click', function() {
        detectButton.disabled = true;
        detectButton.textContent = 'Processing...';
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
            detectedPest = data.pest;
            enableChatFunctionality();
            matchContainerHeights();
            
            // Reset conversation state for new detection
            resetConversation();
        })
        .catch(error => {
            console.error('Error:', error);
            resultContent.innerHTML = `<p>An error occurred while processing the image. Please try again.</p>`;
            detectButton.disabled = false;
            detectButton.textContent = 'Detect Pests';
        });
    });
    
    // Display detection results
    function displayResults(data) {
        resultContainer.style.display = 'block';
        detectButton.disabled = false;
        detectButton.textContent = 'Detect Pests';
        
        if (data.detected) {
            const formattedAdvice = formatManagementAdvice(data.management_advice);
            resultContent.innerHTML = `
                <p><strong>Detected Pest:</strong> ${data.pest}</p>
                <p><strong>Confidence:</strong> ${data.confidence.toFixed(2)}%</p>
                <div class="pest-description">
                    <h3>Description</h3>
                    <p>${data.description}</p>
                </div>
                <div class="management-advice">
                    <h3>Management Advice</h3>
                    ${formattedAdvice}
                </div>
            `;
        } else {
            resultContent.innerHTML = `
                <p>No known pests detected in the image. If you believe there is a pest present, please try a clearer image or a different angle.</p>
            `;
        }
    }
    
    // Format management advice text
    function formatManagementAdvice(advice) {
        if (!advice) return '<p>No management advice available.</p>';
        
        let formatted = '';
        const lines = advice.split('\n');
        
        lines.forEach(line => {
            if (line.trim().startsWith('•') || line.trim().startsWith('-')) {
                const content = line.replace(/^[•-]\s*/, '').trim();
                formatted += `<div class="indented-point">${content}</div>`;
            } else if (line.trim().startsWith('#')) {
                // Handle headings
                const headingLevel = line.match(/^#+/)[0].length;
                const headingText = line.replace(/^#+\s*/, '');
                
                if (headingLevel <= 3) {
                    formatted += `<h${headingLevel}>${headingText}</h${headingLevel}>`;
                } else {
                    formatted += `<strong>${headingText}</strong><br>`;
                }
            } else if (line.trim() === '') {
                formatted += '<br>';
            } else if (line.length > 0) {
                // Regular text
                formatted += `<p>${line}</p>`;
            }
        });
        
        return formatted || advice;
    }
    
    function enableChatFunctionality() {
        chatContainer.style.display = 'block';
        chatInput.focus();
        
        // Start step-by-step conversation
        startStepByStepConversation();
        
        // Ensure chat is scrolled to bottom
        setTimeout(scrollToBottom, 300);
    }
    
    function startStepByStepConversation() {
        // Reset conversation state
        conversationStep = 0;
        userCrop = '';
        managementPreference = '';
        
        // Initial greeting
        appendMessage('ai', `Hello! I see you've detected ${detectedPest}. What crop are you growing?`);
        
        // Update quick responses for first step
        updateQuickResponsesForStep(0);
    }
    
    function updateQuickResponsesForStep(step) {
        quickResponses.innerHTML = '';
        let responses = [];
        
        switch(step) {
            case 0: // What crop are you growing?
                responses = ["Rice", "Wheat", "Corn", "Soybean", "Cotton"];
                break;
            case 1: // Organic or chemical management?
                responses = ["Organic methods", "Chemical methods", "Both", "Preventive measures"];
                break;
            case 2: // Final questions
                responses = [
                    "How can I prevent future infestations?", 
                    "When should I apply these controls?",
                    "Are there resistant crop varieties?",
                    "What's the lifecycle of " + detectedPest + "?"
                ];
                break;
            default:
                responses = [
                    "Tell me more about " + detectedPest,
                    "How do I identify " + detectedPest + "?",
                    "What crops does " + detectedPest + " affect?"
                ];
        }
        
        responses.forEach(response => {
            const button = document.createElement('button');
            button.textContent = response;
            button.classList.add('quick-response-btn');
            button.addEventListener('click', () => sendMessage(response));
            quickResponses.appendChild(button);
        });
    }
    
    // Handle chat functionality
    chatSendButton.addEventListener('click', () => sendMessage());
    
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') sendMessage();
    });
    
    function sendMessage(message = null) {
        const userMessage = message || chatInput.value.trim();
        if (!userMessage) return;
        
        // Save scroll position before adding new message
        lastScrollHeight = chatLog.scrollHeight;
        
        // Display user message
        appendMessage('user', userMessage);
        chatInput.value = '';
        
        // Process step-by-step conversation
        processConversationStep(userMessage);
    }
    
    function processConversationStep(userMessage) {
        switch(conversationStep) {
            case 0: // User provided crop information
                userCrop = userMessage;
                conversationStep = 1;
                
                // Respond and ask for management preference
                setTimeout(() => {
                    appendMessage('ai', `Thank you! You're growing ${userCrop} and dealing with ${detectedPest}. Would you prefer organic or chemical management methods?`);
                    updateQuickResponsesForStep(1);
                }, 500);
                break;
                
            case 1: // User provided management preference
                managementPreference = userMessage;
                conversationStep = 2;
                
                // Fetch specific advice based on preferences
                fetchSpecificAdvice(userCrop, detectedPest, managementPreference);
                break;
                
            default: // General questions after initial flow
                // Send to backend for AI response
                fetchAIResponse(userMessage);
                break;
        }
    }
    
    function fetchSpecificAdvice(crop, pest, preference) {
        // Show typing indicator
        showTypingIndicator();
        
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: `Provide ${preference} management strategies for ${pest} in ${crop} crops.`,
                pest: pest,
                context: {
                    crop: crop,
                    preference: preference
                }
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Display the response
            appendMessage('ai', data.response);
            
            // After providing specific advice, prompt for further questions
            setTimeout(() => {
                appendMessage('ai', "Do you have any specific questions about implementing these methods?");
                updateQuickResponsesForStep(2);
            }, 1000);
        })
        .catch(error => {
            console.error('Error sending chat message:', error);
            removeTypingIndicator();
            appendMessage('ai', "I'm sorry, I couldn't process your request at this time. Please try again.");
        });
    }
    
    function fetchAIResponse(userMessage) {
        // Show typing indicator
        showTypingIndicator();
        
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: userMessage,
                pest: detectedPest,
                context: {
                    crop: userCrop,
                    preference: managementPreference
                }
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Display the response
            appendMessage('ai', data.response);
        })
        .catch(error => {
            console.error('Error sending chat message:', error);
            removeTypingIndicator();
            appendMessage('ai', "I'm sorry, I couldn't process your request at this time. Please try again.");
        });
    }
    
    // Typing indicator functions
    function showTypingIndicator() {
        const typingElement = document.createElement('div');
        typingElement.classList.add('chat-message', 'ai', 'typing-indicator');
        
        const contentElement = document.createElement('div');
        contentElement.classList.add('message-content');
        contentElement.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
        
        typingElement.appendChild(contentElement);
        chatLog.appendChild(typingElement);
        
        scrollToBottom();
    }
    
    function removeTypingIndicator() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    function appendMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender);
        
        const contentElement = document.createElement('div');
        contentElement.classList.add('message-content');
        
        // Remove star symbols and format the message
        const sanitizedMessage = message.replace(/\*/g, '').trim();
        contentElement.textContent = sanitizedMessage;
        
        messageElement.appendChild(contentElement);
        chatLog.appendChild(messageElement);
        if (sender === 'user') {
        messageElement.style.alignSelf = 'flex-end';
        contentElement.style.textAlign = 'right';
    }
        // Check if we should scroll to bottom
        if (isScrolledToBottom) {
            scrollToBottom();
        } else {
            // If user has scrolled up, maintain relative scroll position
            maintainScrollPosition();
            // Show new message indicator
            showNewMessageIndicator();
        }
    }
    
    function showNewMessageIndicator() {
        if (!isScrolledToBottom) {
            scrollToBottomBtn.style.display = 'block';
            
            // Add animation to draw attention
            scrollToBottomBtn.animate([
                { transform: 'scale(1)' },
                { transform: 'scale(1.2)' },
                { transform: 'scale(1)' }
            ], {
                duration: 500,
                iterations: 2
            });
        }
    }
    
    function scrollToBottom() {
        // Use setTimeout to ensure DOM has updated
        setTimeout(() => {
            chatLog.style.scrollBehavior = 'smooth';
            chatLog.scrollTop = chatLog.scrollHeight;
            
            // Reset scroll behavior after animation
            setTimeout(() => {
                chatLog.style.scrollBehavior = 'auto';
                scrollToBottomBtn.style.display = 'none';
            }, 500);
        }, 50);
    }
    
    function maintainScrollPosition() {
        // Calculate how much new content was added
        const newContentHeight = chatLog.scrollHeight - lastScrollHeight;
        
        // Adjust scroll position to maintain the same relative view
        if (newContentHeight > 0) {
            chatLog.scrollTop += newContentHeight;
        }
    }
    
    // Toggle chat visibility
    toggleChat.addEventListener('click', function() {
        const chatBody = chatContainer.querySelector('.chat-body');
        chatBody.style.display = chatBody.style.display === 'none' ? 'flex' : 'none';
        toggleChat.innerHTML = chatBody.style.display === 'none' ? '▼' : '▲';
    });
    
    // Listen for window resize to adjust container heights
    window.addEventListener('resize', function() {
        matchContainerHeights();
        if (isScrolledToBottom) {
            scrollToBottom();
        }
    });
    
    // Add CSS for typing indicator
    const style = document.createElement('style');
    style.textContent = `
        .typing-indicator .message-content {
            padding: 10px 15px;
            display: flex;
            align-items: center;
        }
        
        .typing-indicator .dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #888;
            margin: 0 2px;
            animation: typing-animation 1.4s infinite ease-in-out;
        }
        
        .typing-indicator .dot:nth-child(1) {
            animation-delay: 0s;
        }
        
        .typing-indicator .dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator .dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing-animation {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }
        
        .scroll-to-bottom-btn {
            opacity: 0.8;
            transition: opacity 0.3s, transform 0.3s;
        }
        
        .scroll-to-bottom-btn:hover {
            opacity: 1;
            transform: scale(1.1);
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        
        .new-message-indicator {
            animation: pulse 1s infinite;
        }
    `;
    document.head.appendChild(style);
});
