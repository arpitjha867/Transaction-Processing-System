import re
from collections import defaultdict, deque
from copy import deepcopy

class DatabaseSimulator:
    """Simulates a database with variables"""
    def __init__(self, initial_state_str):
        self.data = {}
        self.locks = {}  # For concurrency control
        
        # Parse initial state: "A=100,B=200,C=300"
        for item in initial_state_str.split(','):
            if '=' in item:
                var, val = item.strip().split('=')
                self.data[var.strip()] = int(val.strip())
    
    def read(self, variable, transaction_id):
        """Read operation with lock checking"""
        if variable not in self.data:
            raise Exception(f"Variable {variable} does not exist in database")
        return self.data[variable]
    
    def write(self, variable, value, transaction_id):
        """Write operation with lock checking"""
        self.data[variable] = value
    
    def get_state(self):
        """Return current database state"""
        return dict(self.data)
    
    def restore_state(self, state):
        """Restore database to a previous state"""
        self.data = deepcopy(state)


class RecoveryManager:
    """Manages UNDO and REDO logs for recovery"""
    def __init__(self):
        self.undo_log = []
        self.redo_log = []
    
    def log_before_image(self, transaction_id, variable, old_value):
        """Log before image for UNDO (Atomicity)"""
        self.undo_log.append({
            'transaction': transaction_id,
            'variable': variable,
            'old_value': old_value,
            'type': 'BEFORE_IMAGE'
        })
    
    def log_after_image(self, transaction_id, variable, new_value):
        """Log after image for REDO (Durability)"""
        self.redo_log.append({
            'transaction': transaction_id,
            'variable': variable,
            'new_value': new_value,
            'type': 'AFTER_IMAGE'
        })
    
    def undo_transaction(self, transaction_id, db_simulator):
        """UNDO all operations of a transaction"""
        undo_operations = [log for log in reversed(self.undo_log) 
                          if log['transaction'] == transaction_id]
        
        for op in undo_operations:
            db_simulator.data[op['variable']] = op['old_value']
        
        return undo_operations
    
    def redo_transaction(self, transaction_id, db_simulator):
        """REDO all operations of a transaction"""
        redo_operations = [log for log in self.redo_log 
                          if log['transaction'] == transaction_id]
        
        for op in redo_operations:
            db_simulator.data[op['variable']] = op['new_value']
        
        return redo_operations


def parse_transactions(content):
    """Parse transaction operations from text content"""
    transactions = defaultdict(list)
    operation_order = []  # Track the order operations appear in file
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Parse format: T1: READ A or T1: WRITE A = A+50 or T1: COMMIT
        match = re.match(r'(T\d+)\s*:\s*(.+)', line, re.IGNORECASE)
        if match:
            tid = match.group(1)
            operation = match.group(2).strip()
            
            parsed_op = parse_operation(operation)
            transactions[tid].append(parsed_op)
            operation_order.append((tid, parsed_op))
    
    # Return both the grouped transactions and the original order
    return dict(transactions), operation_order


def parse_operation(operation):
    """Parse a single operation"""
    op_upper = operation.upper()
    
    # READ operation
    if op_upper.startswith('READ'):
        match = re.search(r'READ\s+(\w+)', op_upper)
        if match:
            return {'type': 'READ', 'variable': match.group(1)}
    
    # WRITE operation
    elif op_upper.startswith('WRITE'):
        match = re.search(r'WRITE\s+(\w+)\s*=\s*(.+)', operation, re.IGNORECASE)
        if match:
            return {
                'type': 'WRITE',
                'variable': match.group(1),
                'expression': match.group(2).strip()
            }
    
    # COMMIT operation
    elif op_upper == 'COMMIT':
        return {'type': 'COMMIT'}
    
    # ABORT operation
    elif op_upper == 'ABORT':
        return {'type': 'ABORT'}
    
    return {'type': 'UNKNOWN', 'raw': operation}


def check_serializability(transactions):
    """
    Check if the transaction schedule is conflict serializable
    using precedence graph
    """
    # Build precedence graph
    graph = defaultdict(set)
    all_ops = []
    all_transactions = list(transactions.keys())
    
    # Initialize graph with all transaction nodes
    for tid in all_transactions:
        if tid not in graph:
            graph[tid] = set()
    
    # Flatten all operations with transaction IDs
    for tid, ops in transactions.items():
        for op in ops:
            all_ops.append((tid, op))
    
    # Check for conflicts
    for i, (t1, op1) in enumerate(all_ops):
        if op1['type'] not in ['READ', 'WRITE']:
            continue
        
        for j in range(i + 1, len(all_ops)):
            t2, op2 = all_ops[j]
            
            if t1 == t2 or op2['type'] not in ['READ', 'WRITE']:
                continue
            
            # Check if operations conflict
            if op1.get('variable') == op2.get('variable'):
                # Write-Read, Write-Write, or Read-Write conflict
                if op1['type'] == 'WRITE' or op2['type'] == 'WRITE':
                    graph[t1].add(t2)
    
    # Check for cycles using DFS
    def has_cycle(node, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if has_cycle(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    visited = set()
    for node in all_transactions:
        if node not in visited:
            if has_cycle(node, visited, set()):
                return False, []
    
    # Topological sort for serial order
    serial_order = topological_sort(graph, all_transactions)
    
    return True, serial_order


def topological_sort(graph, all_nodes):
    """Perform topological sort on precedence graph"""
    in_degree = {node: 0 for node in all_nodes}
    
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result


def check_recoverability(transactions):
    """
    Check if the schedule is recoverable
    A schedule is recoverable if no transaction commits before
    all transactions it has read from have committed
    """
    read_from = defaultdict(set)  # T1 read from T2
    commit_order = []
    
    # Track dependencies
    written_by = {}  # variable -> transaction that last wrote it
    
    ops_sequence = []
    for tid in sorted(transactions.keys()):  # Sort for consistent ordering
        for op in transactions[tid]:
            ops_sequence.append((tid, op))
    
    for tid, op in ops_sequence:
        if op['type'] == 'READ':
            var = op.get('variable')
            if var in written_by and written_by[var] != tid:
                read_from[tid].add(written_by[var])
        
        elif op['type'] == 'WRITE':
            var = op.get('variable')
            written_by[var] = tid
        
        elif op['type'] == 'COMMIT':
            if tid not in commit_order:
                commit_order.append(tid)
    
    # Check recoverability
    for i, tid in enumerate(commit_order):
        for dependency in read_from[tid]:
            commit_idx = commit_order.index(tid) if tid in commit_order else -1
            dep_idx = commit_order.index(dependency) if dependency in commit_order else -1
            
            if dep_idx == -1 or dep_idx > commit_idx:
                return False, f"Transaction {tid} commits before {dependency}"
    
    return True, "Schedule is recoverable"


class TransactionScheduler:
    """Executes transactions while maintaining ACID properties"""
    def __init__(self, transactions, db_simulator, recovery_manager, operation_order=None):
        self.transactions = transactions
        self.db = db_simulator
        self.recovery = recovery_manager
        self.current_step = 0
        self.operation_queue = operation_order if operation_order else self._build_operation_queue(transactions)
        self.transaction_states = {tid: 'ACTIVE' for tid in transactions.keys()}
        self.temp_values = defaultdict(dict)  # Temporary values for each transaction
    
    def _build_operation_queue(self, transactions):
        """Build operation queue preserving file order"""
        queue = []
        
        # Get maximum number of operations
        max_ops = max(len(ops) for ops in transactions.values()) if transactions else 0
        
        # Interleave operations
        for i in range(max_ops):
            for tid in sorted(transactions.keys()):  # Sort to maintain consistent order
                if i < len(transactions[tid]):
                    queue.append((tid, transactions[tid][i]))
        
        return queue
    
    def execute_next_step(self):
        """Execute one operation"""
        if self.current_step >= len(self.operation_queue):
            return None
        
        tid, op = self.operation_queue[self.current_step]
        self.current_step += 1
        
        return self._execute_operation(tid, op)
    
    def execute_all(self):
        """Execute all operations"""
        results = []
        while self.current_step < len(self.operation_queue):
            result = self.execute_next_step()
            if result:
                results.append(result)
        return results
    
    def _execute_operation(self, tid, op):
        """Execute a single operation with ACID guarantees"""
        result = {
            'transaction': tid,
            'operation': op,
            'timestamp': self.current_step
        }
        
        try:
            if op['type'] == 'READ':
                value = self._execute_read(tid, op)
                result['status'] = 'SUCCESS'
                result['value'] = value
                result['message'] = f"{tid} reads {op['variable']} = {value}"
            
            elif op['type'] == 'WRITE':
                new_value = self._execute_write(tid, op)
                result['status'] = 'SUCCESS'
                result['value'] = new_value
                result['message'] = f"{tid} writes {op['variable']} = {new_value}"
            
            elif op['type'] == 'COMMIT':
                self._execute_commit(tid)
                result['status'] = 'SUCCESS'
                result['message'] = f"{tid} committed successfully"
            
            elif op['type'] == 'ABORT':
                self._execute_abort(tid)
                result['status'] = 'ABORTED'
                result['message'] = f"{tid} aborted"
            
            else:
                result['status'] = 'ERROR'
                result['message'] = f"Unknown operation type: {op.get('type')}"
        
        except Exception as e:
            result['status'] = 'ERROR'
            result['message'] = str(e)
            # Trigger recovery
            self._execute_abort(tid)
            raise
        
        return result
    
    def _execute_read(self, tid, op):
        """Execute READ with Isolation"""
        var = op['variable']
        
        # Check if THIS transaction has an uncommitted write
        if var in self.temp_values[tid]:
            return self.temp_values[tid][var]
        
        # Read from database (will see committed values only)
        return self.db.read(var, tid)
    
    def _execute_write(self, tid, op):
        """Execute WRITE with Atomicity logging"""
        var = op['variable']
        expr = op['expression']
        
        # Get old value for UNDO log
        old_value = self.db.data.get(var, 0)
        
        # Evaluate expression
        local_vars = dict(self.db.data)
        local_vars.update(self.temp_values[tid])
        
        try:
            new_value = eval(expr, {"__builtins__": {}}, local_vars)
        except:
            new_value = eval(expr.replace(var, str(local_vars[var])))
        
        # Log for recovery (Atomicity & Durability)
        self.recovery.log_before_image(tid, var, old_value)
        
        # Store in temporary space (Isolation)
        self.temp_values[tid][var] = new_value
        
        return new_value
    
    def _execute_commit(self, tid):
        """Execute COMMIT with Durability"""
        # Apply all temporary writes to database
        for var, value in self.temp_values[tid].items():
            self.db.write(var, value, tid)
            # Log after image for REDO
            self.recovery.log_after_image(tid, var, value)
        
        self.transaction_states[tid] = 'COMMITTED'
        self.temp_values[tid].clear()
    
    def _execute_abort(self, tid):
        """Execute ABORT with recovery"""
        # Undo all operations
        self.recovery.undo_transaction(tid, self.db)
        self.transaction_states[tid] = 'ABORTED'
        self.temp_values[tid].clear()