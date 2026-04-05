# Routine Compiler

The NavIRL routine compiler allows you to specify structured human behaviors as YAML documents and compile them into executable behavior trees. This enables creating complex, realistic human routines for simulation scenarios.

## Quick Start

### Basic Example

Create a simple routine that has an agent go to a location, wait, then go somewhere else:

```yaml
id: "morning_coffee"
description: "Get morning coffee routine"
tasks:
  - type: "go_to"
    params:
      x: -2.0
      y: 1.0
    priority: 1
  - type: "wait"
    params:
      duration: 5.0
    priority: 2
  - type: "go_to"
    params:
      x: 3.0
      y: -1.0
    priority: 3
repetitions: 1
loop: false
```

### Using in a Scenario

To use compiled routines in a scenario, specify them in your scenario YAML:

```yaml
# scenario.yaml
id: office_routine_demo
description: Office workers following daily routines
scene:
  backend: grid2d
  map:
    source: builtin
    id: office
# ... other scenario config ...

humans:
  controller:
    type: compiled_routine
    params:
      routines:
        1:  # Agent ID 1
          id: "morning_coffee"
          description: "Get morning coffee"
          tasks:
            - type: "go_to"
              params: {"x": -2.0, "y": 1.0}
            - type: "wait"
              params: {"duration": 5.0}
        2:  # Agent ID 2
          id: "check_printer"
          description: "Check the printer"
          tasks:
            - type: "go_to"
              params: {"x": 5.0, "y": 2.0}
            - type: "interact"
              params:
                location: [5.0, 2.0]
                interaction_type: "printer"
      fallback_behavior: "goal_swap"
  count: 2
  # ... rest of humans config ...
```

## Routine Schema

### Core Structure

Every routine specification must include:

- **id**: Unique identifier for the routine
- **description**: Human-readable description
- **tasks**: List of tasks to execute

Optional fields:
- **branches**: Conditional behavior branches
- **temporal_constraints**: Time-based constraints
- **repetitions**: Number of times to repeat (default: 1)
- **loop**: Whether to loop indefinitely (default: false)
- **metadata**: Additional custom data

### Task Types

#### go_to
Move to a specific location.

```yaml
- type: "go_to"
  params:
    x: 5.0          # Target X coordinate
    y: 3.0          # Target Y coordinate
    speed: 1.0      # Optional: movement speed
  priority: 1       # Higher numbers = higher priority
```

#### wait
Wait for a specified duration.

```yaml
- type: "wait"
  params:
    duration: 10.0  # Wait time in seconds
  priority: 1
```

#### interact
Interact with an object or location.

```yaml
- type: "interact"
  params:
    location: [2.0, 3.0]           # Interaction location
    interaction_type: "computer"    # Type of interaction
  priority: 1
```

#### queue
Join and wait in a queue.

```yaml
- type: "queue"
  params:
    queue_location: [1.0, 2.0]     # Where to queue
    max_wait_time: 300.0           # Maximum wait time
  priority: 1
```

#### follow
Follow group behavior (maintain cohesion with nearby agents).

```yaml
- type: "follow"
  params:
    group_radius: 3.0              # Maximum distance for group
  priority: 1
```

#### avoid
Avoid a specific area or agent.

```yaml
- type: "avoid"
  params:
    location: [0.0, 0.0]           # Location to avoid
    radius: 2.0                    # Avoidance radius
    # OR
    agent_id: 42                   # Specific agent to avoid
  priority: 1
```

#### group
Maintain group cohesion.

```yaml
- type: "group"
  params:
    max_separation: 2.0            # Maximum separation distance
  priority: 1
```

### Conditional Branches

Add conditional behavior using branches:

```yaml
id: "conditional_routine"
description: "Routine with conditional behavior"
tasks:
  - type: "go_to"
    params: {"x": 1.0, "y": 1.0}
branches:
  - condition:
      type: "time_elapsed"
      params:
        seconds: 30.0
    tasks:
      - type: "go_to"
        params: {"x": 5.0, "y": 5.0}
    probability: 0.8               # 80% chance if condition met
```

#### Condition Types

**time_elapsed**: Trigger after time passes
```yaml
condition:
  type: "time_elapsed"
  params:
    seconds: 60.0
```

**location_reached**: Trigger when agent reaches a location
```yaml
condition:
  type: "location_reached"
  params:
    x: 2.0
    y: 3.0
    radius: 0.5
```

**agent_nearby**: Trigger when another agent is nearby
```yaml
condition:
  type: "agent_nearby"
  params:
    agent_id: 42                   # Optional: specific agent
    distance: 2.0
```

### Temporal Constraints

Control when and how long routines run:

```yaml
temporal_constraints:
  start_time: 10.0                 # Don't start until 10 seconds
  end_time: 120.0                  # Must end by 120 seconds
  max_duration: 60.0               # Maximum duration
  min_duration: 30.0               # Minimum duration
```

### Repetitions and Looping

```yaml
# Repeat exactly 3 times then stop
repetitions: 3
loop: false

# Loop indefinitely
repetitions: -1  # Ignored when loop=true
loop: true

# Repeat 5 times then loop forever
repetitions: 5
loop: true
```

## Advanced Examples

### Daily Office Routine

```yaml
id: "office_worker_daily"
description: "Typical office worker daily routine"
tasks:
  # Arrive at desk
  - type: "go_to"
    params: {"x": 2.0, "y": 3.0}
    priority: 10

  # Check email (simulate with wait)
  - type: "wait"
    params: {"duration": 15.0}
    priority: 9

  # Get coffee
  - type: "go_to"
    params: {"x": -5.0, "y": 1.0}
    priority: 8
  - type: "interact"
    params:
      location: [-5.0, 1.0]
      interaction_type: "coffee_machine"
    priority: 7

  # Return to desk
  - type: "go_to"
    params: {"x": 2.0, "y": 3.0}
    priority: 6

branches:
  # Occasionally go to printer
  - condition:
      type: "time_elapsed"
      params: {"seconds": 45.0}
    tasks:
      - type: "go_to"
        params: {"x": 8.0, "y": -2.0}
      - type: "interact"
        params:
          location: [8.0, -2.0]
          interaction_type: "printer"
    probability: 0.3

  # Sometimes talk to colleagues
  - condition:
      type: "agent_nearby"
      params: {"distance": 1.5}
    tasks:
      - type: "wait"
        params: {"duration": 10.0}
    probability: 0.4

temporal_constraints:
  max_duration: 300.0              # 5 minute maximum

repetitions: 3                     # Do this routine 3 times
loop: false
```

### Restaurant Customer Routine

```yaml
id: "restaurant_customer"
description: "Customer dining experience"
tasks:
  # Enter and wait to be seated
  - type: "go_to"
    params: {"x": 0.0, "y": -3.0}
    priority: 10
  - type: "queue"
    params:
      queue_location: [0.0, -3.0]
      max_wait_time: 120.0
    priority: 9

  # Go to table
  - type: "go_to"
    params: {"x": 3.0, "y": 2.0}
    priority: 8

  # Dining time
  - type: "wait"
    params: {"duration": 45.0}
    priority: 7

  # Go to counter to pay
  - type: "go_to"
    params: {"x": -2.0, "y": -1.0}
    priority: 6
  - type: "interact"
    params:
      location: [-2.0, -1.0]
      interaction_type: "payment"
    priority: 5

  # Exit
  - type: "go_to"
    params: {"x": 0.0, "y": -5.0}
    priority: 4

branches:
  # Sometimes visit restroom
  - condition:
      type: "time_elapsed"
      params: {"seconds": 25.0}
    tasks:
      - type: "go_to"
        params: {"x": -4.0, "y": 3.0}
      - type: "wait"
        params: {"duration": 8.0}
      - type: "go_to"
        params: {"x": 3.0, "y": 2.0}  # Back to table
    probability: 0.25

loop: false
repetitions: 1
metadata:
  customer_type: "casual_diner"
  expected_duration_minutes: 60
```

### Group Shopping Routine

```yaml
id: "group_shopping"
description: "Shopping group routine with coordination"
tasks:
  # Meet at entrance
  - type: "go_to"
    params: {"x": 0.0, "y": 0.0}
    priority: 10

  # Maintain group cohesion while shopping
  - type: "group"
    params: {"max_separation": 2.5}
    priority: 9

  # Visit different store sections
  - type: "go_to"
    params: {"x": 5.0, "y": 3.0}
    priority: 8
  - type: "wait"
    params: {"duration": 20.0}
    priority: 7

  - type: "go_to"
    params: {"x": -3.0, "y": 5.0}
    priority: 6
  - type: "wait"
    params: {"duration": 15.0}
    priority: 5

branches:
  # Split up occasionally
  - condition:
      type: "time_elapsed"
      params: {"seconds": 30.0}
    tasks:
      - type: "go_to"
        params: {"x": 8.0, "y": -2.0}
      - type: "wait"
        params: {"duration": 10.0}
      - type: "group"  # Rejoin group
        params: {"max_separation": 3.0}
    probability: 0.4

temporal_constraints:
  max_duration: 180.0

loop: true                         # Keep shopping
```

## Integration with Scenarios

### Controller Configuration

Use the `compiled_routine` controller type in your scenario:

```yaml
humans:
  controller:
    type: compiled_routine
    params:
      routines:
        # Map agent ID to routine spec
        1: { /* routine spec */ }
        2: { /* routine spec */ }
      fallback_behavior: "goal_swap"  # or "static"
```

### Loading from Files

You can also load routines from separate YAML files:

```python
from navirl.routines import RoutineControllerFactory

# Load from files
controller = RoutineControllerFactory.from_yaml_files({
    1: "routines/office_worker.yaml",
    2: "routines/customer.yaml"
})

# Use in scenario
scenario_config["humans"]["controller"] = {
    "type": "compiled_routine",
    "instance": controller
}
```

## Custom Extensions

### Custom Task Types

Register custom task handlers:

```python
from navirl.routines.compiler import RoutineCompiler
from navirl.models.behavior_tree import ActionNode

class CustomTaskNode(ActionNode):
    def tick(self, bb):
        # Your custom logic
        return Status.SUCCESS

def custom_task_handler(task):
    return CustomTaskNode()

compiler = RoutineCompiler()
compiler.register_custom_task_handler("my_task", custom_task_handler)
```

### Custom Conditions

Register custom condition handlers:

```python
def custom_condition_handler(condition):
    def predicate(bb):
        # Your custom condition logic
        return True  # or False
    return predicate

compiler.register_custom_condition_handler("my_condition", custom_condition_handler)
```

## Best Practices

### 1. Use Priorities Effectively
- Higher priority tasks execute first in sequences
- Use priorities to ensure critical tasks happen before optional ones
- Typical range: 1-10, where 10 is highest priority

### 2. Design for Realism
- Include natural variations with probabilistic branches
- Add appropriate wait times for realistic interactions
- Consider group dynamics and social behaviors

### 3. Handle Edge Cases
- Set reasonable temporal constraints to prevent infinite loops
- Use fallback behaviors for robustness
- Test routines with different agent configurations

### 4. Optimize Performance
- Avoid overly complex branching structures
- Use efficient condition checks
- Consider the computational cost of your routines

### 5. Maintain Readability
- Use descriptive IDs and descriptions
- Comment complex logic in metadata
- Keep individual routines focused and modular

## Debugging and Troubleshooting

### Common Issues

**Routine Not Executing**
- Check that agent ID matches routine assignment
- Verify routine compiles without errors
- Ensure scenario controller type is set correctly

**Unexpected Behavior**
- Check task priorities and execution order
- Verify condition logic and parameters
- Review temporal constraints

**Performance Issues**
- Simplify complex branching logic
- Reduce number of simultaneous condition checks
- Optimize custom handlers

### Event Monitoring

The routine controller emits events for monitoring:

```python
def event_handler(event_type, agent_id, data):
    if event_type == "routine_completed":
        print(f"Agent {agent_id} completed routine {data['routine_id']}")
    elif event_type == "routine_error":
        print(f"Error in routine for agent {agent_id}: {data['error']}")

# Events: routine_completed, routine_loop, routine_error
```

## Examples Repository

See the `navirl/scenarios/library/` directory for complete scenario examples using compiled routines:

- `office_daily_routines.yaml` - Office workers following daily patterns
- `restaurant_service.yaml` - Restaurant customers and staff coordination
- `shopping_mall_crowds.yaml` - Complex multi-group shopping behaviors

## Schema Reference

The complete JSON schema for routine specifications is available in `navirl/routines/schema.py`. Use it for validation and IDE integration with YAML editing tools.
