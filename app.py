from flask import Flask, render_template, request, jsonify
import os
from helpers import (
    parse_transactions,
    check_serializability,
    check_recoverability,
    TransactionScheduler,
    DatabaseSimulator,
    RecoveryManager
)

app = Flask(__name__)

# Global objects
db_simulator = None
scheduler = None
recovery_manager = None
transaction_log = []
execution_log = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global db_simulator, scheduler, recovery_manager, transaction_log, execution_log
    
    try:
        # Reset state
        transaction_log = []
        execution_log = []
        
        # Get text input
        content = request.form.get('transaction_text', '')
        
        if not content.strip():
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Get initial database state
        db_state = request.form.get('db_state', 'A=100,B=200,C=300')
        
        # Initialize database
        db_simulator = DatabaseSimulator(db_state)
        recovery_manager = RecoveryManager()
        
        # Parse transactions
        transactions, operation_order = parse_transactions(content)
        if not transactions:
            return jsonify({'error': 'No valid transactions found'}), 400
        
        # Check serializability
        is_serializable, serial_order = check_serializability(transactions)
        
        # Check recoverability
        is_recoverable, recovery_info = check_recoverability(transactions)
        
        # Initialize scheduler with operation order
        scheduler = TransactionScheduler(transactions, db_simulator, recovery_manager, operation_order)
        
        return jsonify({
            'success': True,
            'transactions': transactions,
            'is_serializable': is_serializable,
            'serial_order': serial_order,
            'is_recoverable': is_recoverable,
            'recovery_info': recovery_info,
            'initial_db_state': db_simulator.get_state()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/execute', methods=['POST'])
def execute_transactions():
    global scheduler, execution_log
    
    try:
        if not scheduler:
            return jsonify({'error': 'No transactions loaded'}), 400
        
        # Execute all operations
        execution_log = scheduler.execute_all()
        
        return jsonify({
            'success': True,
            'execution_log': execution_log,
            'final_db_state': db_simulator.get_state(),
            'undo_log': recovery_manager.undo_log,
            'redo_log': recovery_manager.redo_log
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'execution_log': execution_log,
            'recovery_triggered': True,
            'undo_log': recovery_manager.undo_log,
            'redo_log': recovery_manager.redo_log
        })

@app.route('/step', methods=['POST'])
def execute_step():
    global scheduler, execution_log
    
    try:
        if not scheduler:
            return jsonify({'error': 'No transactions loaded'}), 400
        
        step_result = scheduler.execute_next_step()
        
        if step_result:
            execution_log.append(step_result)
            
            return jsonify({
                'success': True,
                'step': step_result,
                'current_db_state': db_simulator.get_state(),
                'undo_log': recovery_manager.undo_log,
                'redo_log': recovery_manager.redo_log,
                'completed': False
            })
        else:
            return jsonify({
                'success': True,
                'completed': True,
                'final_db_state': db_simulator.get_state(),
                'undo_log': recovery_manager.undo_log,
                'redo_log': recovery_manager.redo_log
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'recovery_triggered': True,
            'undo_log': recovery_manager.undo_log,
            'redo_log': recovery_manager.redo_log
        })

@app.route('/reset', methods=['POST'])
def reset():
    global db_simulator, scheduler, recovery_manager, transaction_log, execution_log
    
    db_simulator = None
    scheduler = None
    recovery_manager = None
    transaction_log = []
    execution_log = []
    
    return jsonify({'success': True})

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, port=5000)